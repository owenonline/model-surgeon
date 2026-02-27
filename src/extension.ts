import * as vscode from 'vscode';
import * as path from 'path';
import { parseHeader } from './parsers/headerParser';
import { loadShardedModel, isShardedModel } from './parsers/shardedModel';
import { detectLoraAdapters, parseAdapterConfig } from './parsers/loraDetector';
import { buildArchitectureTree } from './parsers/treeBuilder';
import { alignArchitectures } from './parsers/alignment';
import { WorkerPool } from './workers/workerPool';
import { MessageHost } from './protocol/messageHost';
import { PROTOCOL_VERSION } from './types/messages';

let workerPool: WorkerPool | undefined;
let currentPanel: vscode.WebviewPanel | undefined;
let currentMessageHost: MessageHost | undefined;

let currentModelA: {
  filePath: string;
  tree: import('./types/tree').ArchitectureNode;
  loraMap: import('./types/lora').LoraAdapterMap;
  tensors: Record<string, import('./types/safetensors').TensorInfo>;
} | undefined;

export function activate(context: vscode.ExtensionContext) {
  workerPool = new WorkerPool(2);

  const openModelCmd = vscode.commands.registerCommand(
    'modelSurgeon.openModel',
    async () => {
      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: {
          'Safetensors': ['safetensors'],
          'Index JSON': ['json'],
        },
        title: 'Open Safetensors Model',
      });

      if (!uris || uris.length === 0) return;
      await openModel(uris[0].fsPath, context);
    },
  );

  const openComparisonCmd = vscode.commands.registerCommand(
    'modelSurgeon.openComparison',
    async () => {
      if (!currentPanel) {
        vscode.window.showErrorMessage('Open a model first before comparing.');
        return;
      }

      const uris = await vscode.window.showOpenDialog({
        canSelectMany: false,
        filters: {
          'Safetensors': ['safetensors'],
          'Index JSON': ['json'],
        },
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

  const saveSurgeryCmd = vscode.commands.registerCommand(
    'modelSurgeon.saveSurgeryResult',
    async () => {
      vscode.window.showInformationMessage('Surgery save not yet implemented.');
    },
  );

  context.subscriptions.push(openModelCmd, openComparisonCmd, saveSurgeryCmd);

  // Activate on .safetensors file open
  vscode.workspace.onDidOpenTextDocument((doc) => {
    if (doc.fileName.endsWith('.safetensors')) {
      openModel(doc.fileName, context);
    }
  });
}

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

  currentPanel.webview.html = getWebviewContent(currentPanel.webview, context);
  messageHost.attach(currentPanel);
  currentMessageHost = messageHost;

  // Register handlers for incoming messages
  messageHost.onMessage('loadModel', async (msg) => {
    await loadAndSendModel(msg.filePath, messageHost);
  });

  await loadAndSendModel(filePath, messageHost);
}

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
    let tensors: Record<string, import('./types/safetensors').TensorInfo>;

    if (sharded) {
      const unified = await loadShardedModel(filePath);
      tensors = unified.tensors;
    } else {
      const header = await parseHeader(filePath);
      tensors = header.tensors;
    }

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

    currentModelA = { filePath, tree, loraMap, tensors };

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
    messageHost.postMessage({
      type: 'error',
      protocolVersion: PROTOCOL_VERSION,
      message: `Failed to load model: ${message}`,
      code: 'LOAD_ERROR',
    });
  }
}

async function loadAndCompareModel(filePath: string, messageHost: MessageHost) {
  if (!currentModelA) {
    return;
  }

  try {
    messageHost.postMessage({
      type: 'progress',
      protocolVersion: PROTOCOL_VERSION,
      taskId: 'load-compare',
      label: 'Parsing Model B header',
      percent: 10,
    });

    const sharded = await isShardedModel(filePath);
    let tensors: Record<string, import('./types/safetensors').TensorInfo>;

    if (sharded) {
      const unified = await loadShardedModel(filePath);
      tensors = unified.tensors;
    } else {
      const header = await parseHeader(filePath);
      tensors = header.tensors;
    }

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
      type: 'comparisonResult',
      protocolVersion: PROTOCOL_VERSION,
      alignedComponents,
      treeB,
      loraMapB: loraMap,
      filePathB: filePath
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err);
    messageHost.postMessage({
      type: 'error',
      protocolVersion: PROTOCOL_VERSION,
      message: `Failed to load comparison model: ${message}`,
      code: 'LOAD_ERROR',
    });
  }
}

function getWebviewContent(webview: vscode.Webview, context: vscode.ExtensionContext): string {
  const scriptUri = webview.asWebviewUri(
    vscode.Uri.file(path.join(context.extensionPath, 'dist', 'webview', 'main.js')),
  );

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; script-src ${webview.cspSource}; style-src ${webview.cspSource} 'unsafe-inline';">
  <title>Model Surgeon</title>
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
