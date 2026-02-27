import * as vscode from 'vscode';
import * as path from 'path';
import { parseHeader } from './parsers/headerParser';
import { loadShardedModel, isShardedModel } from './parsers/shardedModel';
import { detectLoraAdapters, parseAdapterConfig } from './parsers/loraDetector';
import { buildArchitectureTree } from './parsers/treeBuilder';
import { alignArchitectures } from './parsers/alignment';
import { computeTensorDiffs, readSingleTensorAsFloat32, computeDiffMetrics } from './parsers/tensorDiff';
import { WorkerPool } from './workers/workerPool';
import { MessageHost } from './protocol/messageHost';
import { PROTOCOL_VERSION } from './types/messages';
import { SurgerySession } from './surgery/SurgerySession';
import { UnifiedTensorMap } from './types/safetensors';
import { Logger } from './utils/logger';

let workerPool: WorkerPool | undefined;
let currentPanel: vscode.WebviewPanel | undefined;
let currentMessageHost: MessageHost | undefined;
let currentSurgerySession: SurgerySession | undefined;

// URI stored by "Select for Model Surgeon Compare"
let selectedForCompareUri: vscode.Uri | undefined;

let currentModelA: {
  filePath: string;
  tree: import('./types/tree').ArchitectureNode;
  loraMap: import('./types/lora').LoraAdapterMap;
  tensors: Record<string, import('./types/safetensors').ShardedTensorInfo>;
  shardHeaderLengths: Record<string, number>;
} | undefined;

let currentModelB: {
  tensors: Record<string, import('./types/safetensors').ShardedTensorInfo>;
  shardHeaderLengths: Record<string, number>;
} | undefined;

// ─── Custom Editor Provider ───────────────────────────────────────────────────

class SafetensorsEditorProvider implements vscode.CustomReadonlyEditorProvider {
  constructor(private readonly context: vscode.ExtensionContext) {}

  openCustomDocument(uri: vscode.Uri): vscode.CustomDocument {
    return { uri, dispose: () => {} };
  }

  async resolveCustomEditor(
    document: vscode.CustomDocument,
    webviewPanel: vscode.WebviewPanel,
    _token: vscode.CancellationToken,
  ): Promise<void> {
    webviewPanel.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.file(path.join(this.context.extensionPath, 'dist', 'webview')),
      ],
    };

    // Tear down any existing panel state cleanly
    currentMessageHost?.dispose();

    currentPanel = webviewPanel;
    const messageHost = new MessageHost();
    setupMessageHandlers(messageHost, this.context);
    messageHost.attach(webviewPanel);
    currentMessageHost = messageHost;

    webviewPanel.webview.html = getWebviewContent(webviewPanel.webview, this.context);

    webviewPanel.onDidDispose(() => {
      if (currentPanel === webviewPanel) {
        currentPanel = undefined;
        currentMessageHost = undefined;
        currentModelA = undefined;
      }
      messageHost.dispose();
    });

    await loadAndSendModel(document.uri.fsPath, messageHost);
  }
}

// ─── Shared panel setup helpers ───────────────────────────────────────────────

/**
 * Create or reuse the Model Surgeon webview panel for non-custom-editor usage
 * (e.g. the command palette "Open Model" command or opening an index.json).
 */
async function openModel(filePath: string, context: vscode.ExtensionContext) {
  const messageHost = new MessageHost();

  if (!currentPanel) {
    currentPanel = vscode.window.createWebviewPanel(
      'modelSurgeon',
      'Model Surgeon',
      vscode.ViewColumn.One,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [
          vscode.Uri.file(path.join(context.extensionPath, 'dist', 'webview')),
        ],
      },
    );

    currentPanel.onDidDispose(() => {
      currentPanel = undefined;
      currentMessageHost = undefined;
      currentModelA = undefined;
      messageHost.dispose();
    });
  }

  currentMessageHost?.dispose();
  currentPanel.webview.html = getWebviewContent(currentPanel.webview, context);
  setupMessageHandlers(messageHost, context);
  messageHost.attach(currentPanel);
  currentMessageHost = messageHost;

  await loadAndSendModel(filePath, messageHost);
}

function setupMessageHandlers(messageHost: MessageHost, _context: vscode.ExtensionContext) {
  messageHost.onMessage('loadModel', async (msg) => {
    await loadAndSendModel(msg.filePath, messageHost);
  });

  messageHost.onMessage('loadComparison', async (msg) => {
    if (currentModelA) {
      await loadAndCompareModel(msg.filePath, messageHost);
    }
  });

  // ── On-demand tensor diff ──────────────────────────────────────────────────
  messageHost.onMessage('requestTensorDiff', async (msg) => {
    const { path: tensorPath } = msg;
    try {
      if (!currentModelA || !currentModelB) {
        throw new Error('Both models must be loaded to compute a diff');
      }
      const [floatsA, floatsB] = await Promise.all([
        readSingleTensorAsFloat32(currentModelA.tensors, currentModelA.shardHeaderLengths, tensorPath),
        readSingleTensorAsFloat32(currentModelB.tensors, currentModelB.shardHeaderLengths, tensorPath),
      ]);
      if (!floatsA || !floatsB) {
        throw new Error(
          `Tensor "${tensorPath}" not found in ${!floatsA ? 'Model A' : 'Model B'}. ` +
          `Model A has ${Object.keys(currentModelA.tensors).length} tensors, ` +
          `Model B has ${Object.keys(currentModelB.tensors).length} tensors.`,
        );
      }
      const metrics = computeDiffMetrics(floatsA, floatsB);
      const PREVIEW_N = 20;
      const infoA = currentModelA.tensors[tensorPath];
      messageHost.postMessage({
        type: 'tensorDiffResult',
        protocolVersion: PROTOCOL_VERSION,
        path: tensorPath,
        metrics,
        previewA: Array.from(floatsA.slice(0, PREVIEW_N)),
        previewB: Array.from(floatsB.slice(0, PREVIEW_N)),
        shape: infoA?.shape ?? [],
      });
    } catch (err) {
      messageHost.postMessage({
        type: 'tensorDiffResult',
        protocolVersion: PROTOCOL_VERSION,
        path: tensorPath,
        metrics: null,
        previewA: [],
        previewB: [],
        shape: [],
        error: err instanceof Error ? err.message : String(err),
      });
    }
  });

  // ── On-demand module diff (batch) ──────────────────────────────────────────
  messageHost.onMessage('requestModuleDiff', async (msg) => {
    const { paths } = msg;
    if (!currentModelA || !currentModelB) {
      messageHost.postMessage({
        type: 'moduleDiffResult',
        protocolVersion: PROTOCOL_VERSION,
        results: paths.map((p) => ({ path: p, metrics: null, error: 'Models not loaded' })),
      });
      return;
    }
    const modelA = currentModelA;
    const modelB = currentModelB;
    const results = await Promise.all(
      paths.map(async (tensorPath) => {
        try {
          const [floatsA, floatsB] = await Promise.all([
            readSingleTensorAsFloat32(modelA.tensors, modelA.shardHeaderLengths, tensorPath),
            readSingleTensorAsFloat32(modelB.tensors, modelB.shardHeaderLengths, tensorPath),
          ]);
          if (!floatsA || !floatsB) {
            return { path: tensorPath, metrics: null as null, error: 'Tensor not found in one or both models' };
          }
          return { path: tensorPath, metrics: computeDiffMetrics(floatsA, floatsB) };
        } catch (err) {
          return { path: tensorPath, metrics: null as null, error: err instanceof Error ? err.message : String(err) };
        }
      }),
    );
    messageHost.postMessage({
      type: 'moduleDiffResult',
      protocolVersion: PROTOCOL_VERSION,
      results,
    });
  });

  messageHost.onMessage('performSurgery', async (msg) => {
    if (!currentSurgerySession) {
      messageHost.postMessage({
        type: 'surgeryResult',
        protocolVersion: PROTOCOL_VERSION,
        success: false,
        error: 'No active surgery session',
      });
      return;
    }

    try {
      const op = msg.operation;
      switch (op.operationType) {
        case 'renameTensor':
          if (!op.newName) throw new Error('newName required for renameTensor');
          currentSurgerySession.renameComponent(op.targetPath, op.newName);
          break;
        case 'removeTensor':
          currentSurgerySession.removeTensor(op.targetPath);
          break;
        case 'renameLoraAdapter':
          if (!op.newName) throw new Error('newName required for renameLoraAdapter');
          currentSurgerySession.renameLoraAdapter(op.targetPath, op.newName);
          break;
        case 'removeLoraAdapter':
          currentSurgerySession.removeLoraAdapter(op.targetPath);
          break;
        case 'replaceTensor':
          if (op.sourceModel === 'B' && currentModelB) {
            currentSurgerySession.replaceComponent(op.targetPath, currentModelB.tensors, currentModelB.shardHeaderLengths);
          } else {
            throw new Error('Source model B not available');
          }
          break;
        default:
          throw new Error(`Unknown operation type: ${op.operationType}`);
      }

      const currentState = currentSurgerySession.getCurrentState();
      const newLoraMap = detectLoraAdapters(currentState.tensors, null);
      const newTree = buildArchitectureTree(currentState.tensors, newLoraMap);

      messageHost.postMessage({
        type: 'surgeryResult',
        protocolVersion: PROTOCOL_VERSION,
        success: true,
        updatedTree: newTree,
      });
    } catch (err: unknown) {
      Logger.error(`Surgery operation failed: ${msg.operation?.operationType}`, err);
      messageHost.postMessage({
        type: 'surgeryResult',
        protocolVersion: PROTOCOL_VERSION,
        success: false,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  });
}

// ─── activate ─────────────────────────────────────────────────────────────────

export function activate(context: vscode.ExtensionContext) {
  Logger.initialize();
  Logger.log('Model Surgeon activated', 'Lifecycle');

  workerPool = new WorkerPool(2);

  // Register Custom Editor Provider for .safetensors files
  const editorProvider = new SafetensorsEditorProvider(context);
  context.subscriptions.push(
    vscode.window.registerCustomEditorProvider(
      'modelSurgeon.safetensorsEditor',
      editorProvider,
      {
        webviewOptions: { retainContextWhenHidden: true },
        supportsMultipleEditorsPerDocument: false,
      },
    ),
  );

  // ── Command: Open Model (file picker) ──────────────────────────────────────
  const openModelCmd = vscode.commands.registerCommand(
    'modelSurgeon.openModel',
    async () => {
      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: {
          'Safetensors / Index': ['safetensors', 'json'],
        },
        title: 'Open Safetensors Model',
      });
      if (!uris || uris.length === 0) return;
      await openModel(uris[0].fsPath, context);
    },
  );

  // ── Command: Open Model from URI (explorer context menu) ──────────────────
  const openModelFromUriCmd = vscode.commands.registerCommand(
    'modelSurgeon.openModelFromUri',
    async (uri: vscode.Uri) => {
      if (!uri) return;
      // .safetensors files are handled by the custom editor, but the user may
      // invoke this command explicitly to reopen in a non-custom-editor panel.
      await openModel(uri.fsPath, context);
    },
  );

  // ── Command: Open Comparison (file picker) ─────────────────────────────────
  const openComparisonCmd = vscode.commands.registerCommand(
    'modelSurgeon.openComparison',
    async () => {
      if (!currentPanel) {
        vscode.window.showErrorMessage('Open a model first before comparing.');
        return;
      }
      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: { 'Safetensors / Index': ['safetensors', 'json'] },
        title: 'Open Second Model for Comparison',
      });
      if (!uris || uris.length === 0) return;
      if (currentMessageHost && currentModelA) {
        await loadAndCompareModel(uris[0].fsPath, currentMessageHost);
      } else {
        vscode.window.showErrorMessage('Model A is not fully loaded yet.');
      }
    },
  );

  // ── Command: Save Surgery Result ───────────────────────────────────────────
  const saveSurgeryCmd = vscode.commands.registerCommand(
    'modelSurgeon.saveSurgeryResult',
    async () => {
      vscode.window.showInformationMessage('Surgery save not yet implemented.');
    },
  );

  // ── Command: Select for Model Surgeon Compare ─────────────────────────────
  const selectForCompareCmd = vscode.commands.registerCommand(
    'modelSurgeon.selectForCompare',
    async (uri: vscode.Uri) => {
      if (!uri) return;
      selectedForCompareUri = uri;
      // Set context key so "Compare with Selected" becomes visible
      vscode.commands.executeCommand('setContext', 'modelSurgeon.hasSelectedForCompare', true);
      const label = path.basename(uri.fsPath);
      vscode.window.setStatusBarMessage(`Model Surgeon: selected "${label}" for compare`, 5000);
    },
  );

  // ── Command: Compare with Selected ────────────────────────────────────────
  const compareWithSelectedCmd = vscode.commands.registerCommand(
    'modelSurgeon.compareWithSelected',
    async (uri: vscode.Uri) => {
      if (!uri || !selectedForCompareUri) return;

      const targetPath = uri.fsPath;
      const selectedPath = selectedForCompareUri.fsPath;

      // Decide which model is A and which is B.
      // Convention: the one right-clicked is primary (Model A); selected is compared (Model B).
      // If the right-clicked file isn't already loaded as Model A, load it first.
      // Both calls are sequential so Model A will be ready before comparison starts.
      if (!currentModelA || currentModelA.filePath !== targetPath) {
        await openModel(targetPath, context);
      }

      if (currentMessageHost && currentModelA) {
        await loadAndCompareModel(selectedPath, currentMessageHost);
      }
    },
  );

  context.subscriptions.push(
    openModelCmd,
    openModelFromUriCmd,
    openComparisonCmd,
    saveSurgeryCmd,
    selectForCompareCmd,
    compareWithSelectedCmd,
  );
}

// ─── Model loading ────────────────────────────────────────────────────────────

async function loadAndSendModel(filePath: string, messageHost: MessageHost) {
  try {
    messageHost.postMessage({
      type: 'progress',
      protocolVersion: PROTOCOL_VERSION,
      taskId: 'load',
      label: 'Parsing model header',
      percent: 10,
    });

    const sharded = await isShardedModel(filePath);
    let tensors: Record<string, import('./types/safetensors').ShardedTensorInfo>;
    let unifiedModel: UnifiedTensorMap;

    if (sharded) {
      unifiedModel = await loadShardedModel(filePath);
      tensors = unifiedModel.tensors;
    } else {
      const header = await parseHeader(filePath);
      tensors = Object.fromEntries(
        Object.entries(header.tensors).map(([k, v]) => [k, { ...v, shardFile: filePath }])
      );
      unifiedModel = {
        metadata: header.metadata,
        tensors,
        shardHeaderLengths: { [filePath]: header.headerLength }
      };
    }

    currentSurgerySession = new SurgerySession(unifiedModel);

    messageHost.postMessage({
      type: 'progress',
      protocolVersion: PROTOCOL_VERSION,
      taskId: 'load',
      label: 'Detecting LoRA adapters',
      percent: 50,
    });

    const adapterConfig = await parseAdapterConfig(filePath);
    const loraMap = detectLoraAdapters(tensors, adapterConfig);

    messageHost.postMessage({
      type: 'progress',
      protocolVersion: PROTOCOL_VERSION,
      taskId: 'load',
      label: 'Building architecture tree',
      percent: 75,
    });

    const tree = buildArchitectureTree(tensors, loraMap);

    currentModelA = { filePath, tree, loraMap, tensors, shardHeaderLengths: unifiedModel.shardHeaderLengths };

    messageHost.postMessage({
      type: 'modelLoaded',
      protocolVersion: PROTOCOL_VERSION,
      tree,
      loraMap,
      tensorCount: Object.keys(tensors).length,
      filePath,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    Logger.error(`Failed to load model from ${filePath}`, err);
    vscode.window.showErrorMessage(`Model Surgeon: Failed to load model from ${filePath}. ${message}`);
    messageHost.postMessage({
      type: 'error',
      protocolVersion: PROTOCOL_VERSION,
      message: `Failed to load model: ${message}`,
      code: 'LOAD_ERROR',
    });
  }
}

async function loadAndCompareModel(filePath: string, messageHost: MessageHost) {
  if (!currentModelA) return;

  try {
    messageHost.postMessage({
      type: 'progress',
      protocolVersion: PROTOCOL_VERSION,
      taskId: 'load-compare',
      label: 'Parsing Model B header',
      percent: 10,
    });

    const sharded = await isShardedModel(filePath);
    let tensors: Record<string, import('./types/safetensors').ShardedTensorInfo>;
    let shardHeaderLengths: Record<string, number>;

    if (sharded) {
      const unified = await loadShardedModel(filePath);
      tensors = unified.tensors;
      shardHeaderLengths = unified.shardHeaderLengths;
    } else {
      const header = await parseHeader(filePath);
      tensors = Object.fromEntries(
        Object.entries(header.tensors).map(([k, v]) => [k, { ...v, shardFile: filePath }])
      );
      shardHeaderLengths = { [filePath]: header.headerLength };
    }

    currentModelB = { tensors, shardHeaderLengths };

    messageHost.postMessage({
      type: 'progress',
      protocolVersion: PROTOCOL_VERSION,
      taskId: 'load-compare',
      label: 'Detecting Model B LoRA adapters',
      percent: 50,
    });

    const adapterConfig = await parseAdapterConfig(filePath);
    const loraMap = detectLoraAdapters(tensors, adapterConfig);

    messageHost.postMessage({
      type: 'progress',
      protocolVersion: PROTOCOL_VERSION,
      taskId: 'load-compare',
      label: 'Aligning architectures',
      percent: 75,
    });

    const treeB = buildArchitectureTree(tensors, loraMap);
    const alignedComponents = alignArchitectures(currentModelA.tree, treeB);

    messageHost.postMessage({
      type: 'progress',
      protocolVersion: PROTOCOL_VERSION,
      taskId: 'load-compare',
      label: 'Computing weight diffs',
      percent: 90,
    });

    // Collect paths of matched parameter nodes so we can compute actual diffs.
    const matchedParamPaths = alignedComponents
      .filter((c) => c.status === 'matched' && !c.shapeMismatch)
      .map((c) => c.path);

    const diffMap = await computeTensorDiffs(
      currentModelA.tensors,
      currentModelA.shardHeaderLengths,
      tensors,
      shardHeaderLengths,
      matchedParamPaths,
    );

    // Attach diff metrics to the alignment entries in-place.
    for (const entry of alignedComponents) {
      const metrics = diffMap.get(entry.path);
      if (metrics) entry.diffMetrics = metrics;
    }

    messageHost.postMessage({
      type: 'comparisonResult',
      protocolVersion: PROTOCOL_VERSION,
      alignedComponents,
      treeB,
      loraMapB: loraMap,
      filePathB: filePath
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    Logger.error(`Failed to load comparison model from ${filePath}`, err);
    vscode.window.showErrorMessage(`Model Surgeon: Failed to load comparison model from ${filePath}. ${message}`);
    messageHost.postMessage({
      type: 'error',
      protocolVersion: PROTOCOL_VERSION,
      message: `Failed to load comparison model: ${message}`,
      code: 'LOAD_ERROR',
    });
  }
}

// ─── Webview HTML ─────────────────────────────────────────────────────────────

function getWebviewContent(webview: vscode.Webview, context: vscode.ExtensionContext): string {
  const scriptUri = webview.asWebviewUri(
    vscode.Uri.file(path.join(context.extensionPath, 'dist', 'webview', 'main.js')),
  );
  const cssUri = webview.asWebviewUri(
    vscode.Uri.file(path.join(context.extensionPath, 'dist', 'webview', 'main.css')),
  );

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; script-src ${webview.cspSource}; style-src ${webview.cspSource} 'unsafe-inline';">
  <title>Model Surgeon</title>
  <link rel="stylesheet" href="${cssUri}">
  <style>
    html, body, #root {
      margin: 0;
      padding: 0;
      width: 100vw;
      height: 100vh;
      overflow: hidden;
      background-color: var(--vscode-editor-background);
      color: var(--vscode-editor-foreground);
      font-family: var(--vscode-font-family);
    }
  </style>
</head>
<body>
  <div id="root"></div>
  <script type="module" src="${scriptUri}"></script>
</body>
</html>`;
}

export function deactivate() {
  workerPool?.terminate();
  currentPanel?.dispose();
}
