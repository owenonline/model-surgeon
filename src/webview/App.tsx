import React, { useState } from 'react';
import { useMessage } from './hooks/useMessage';
import { ArchitectureNode } from '../types/tree';
import { LoraAdapterMap } from '../types/lora';
import { SequentialView } from './components/SequentialView';

export function App() {
  const [tree, setTree] = useState<ArchitectureNode | null>(null);
  const [loraMap, setLoraMap] = useState<LoraAdapterMap>({});
  const [filePathA, setFilePathA] = useState<string | null>(null);
  const [comparison, setComparison] = useState<{
    treeB: ArchitectureNode;
    loraMapB: LoraAdapterMap;
    alignedComponents: import('../types/messages').AlignedComponent[];
    filePathB: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<{ label: string; percent: number } | null>(null);

  useMessage({
    modelLoaded: (msg) => {
      setTree(msg.tree);
      setLoraMap(msg.loraMap);
      setFilePathA(msg.filePath);
      setLoading(false);
      setProgress(null);
      setError(null);
    },
    error: (msg) => {
      setError(msg.message);
      setLoading(false);
    },
    progress: (msg) => {
      setProgress({ label: msg.label, percent: msg.percent });
    },
    comparisonResult: (msg) => {
      setComparison({
        treeB: msg.treeB,
        loraMapB: msg.loraMapB,
        alignedComponents: msg.alignedComponents,
        filePathB: msg.filePathB,
      });
      setLoading(false);
      setProgress(null);
    },
    surgeryResult: (msg) => {
      if (msg.updatedTree) {
        setTree(msg.updatedTree);
      }
      if (msg.error) {
        setError(msg.error);
      }
      setLoading(false);
      setProgress(null);
    },
  });

  const handleLoadStats = (node: ArchitectureNode) => {
    // Send a message to the extension to load stats
    // vscode API is handled by the hook/host, we would add a message type for it
  };

  if (error) {
    return (
      <div style={{ padding: '16px', fontFamily: 'var(--vscode-font-family)', color: 'var(--vscode-errorForeground)' }}>
        {error}
      </div>
    );
  }

  if (loading) {
    return <div style={{ padding: '16px' }}>Loading...</div>;
  }

  if (progress) {
    return (
      <div style={{ padding: '16px', fontFamily: 'var(--vscode-font-family)', color: 'var(--vscode-foreground)' }}>
        <p>{progress.label}</p>
        <div style={{
          width: '100%',
          height: '10px',
          backgroundColor: 'var(--vscode-progressBar-background)',
          borderRadius: '5px',
          overflow: 'hidden'
        }}>
          <div style={{
            width: `${Math.max(0, Math.min(100, progress.percent))}%`,
            height: '100%',
            backgroundColor: 'var(--vscode-button-background)',
            transition: 'width 0.3s ease'
          }} />
        </div>
      </div>
    );
  }

  if (tree) {
    return (
      <div style={{ width: '100vw', height: '100vh', margin: 0, padding: 0 }}>
        <SequentialView
          tree={tree}
          loraMap={loraMap}
          filePathA={filePathA || undefined}
          comparison={comparison || undefined}
          onLoadStats={handleLoadStats}
        />
      </div>
    );
  }

  return (
    <div style={{ padding: '32px', fontFamily: 'var(--vscode-font-family)', color: 'var(--vscode-foreground)', maxWidth: '600px', margin: '0 auto' }}>
      <h1 style={{ fontSize: '24px', marginBottom: '16px', fontWeight: 'bold' }}>Welcome to Model Surgeon</h1>
      <p style={{ fontSize: '14px', marginBottom: '24px', lineHeight: '1.5' }}>
        Visualize, compare, and surgically modify neural network models in safetensors format.
      </p>
      
      <div style={{ marginBottom: '32px' }}>
        <h2 style={{ fontSize: '18px', marginBottom: '12px', borderBottom: '1px solid var(--vscode-panel-border)', paddingBottom: '8px' }}>Get Started</h2>
        <ol style={{ paddingLeft: '24px', fontSize: '14px', lineHeight: '1.8' }}>
          <li>Open a <code style={{ backgroundColor: 'var(--vscode-textCodeBlock-background)', padding: '2px 4px', borderRadius: '3px' }}>.safetensors</code> file from your workspace, or use the <strong>Model Surgeon: Open Model</strong> command.</li>
          <li>Explore the architecture graph, expand/collapse layers, and inspect tensors.</li>
          <li>Load a second model via <strong>Model Surgeon: Open Comparison</strong> to see weight differences side-by-side.</li>
          <li>Right-click nodes to perform surgeries (rename, remove, replace).</li>
          <li>Save your changes to a new file via <strong>Model Surgeon: Save Surgery Result</strong>.</li>
        </ol>
      </div>

      <div style={{ padding: '16px', backgroundColor: 'var(--vscode-textBlockQuote-background)', borderLeft: '4px solid var(--vscode-textBlockQuote-border)', borderRadius: '2px' }}>
        <h3 style={{ fontSize: '14px', margin: '0 0 8px 0' }}>Keyboard Shortcuts</h3>
        <ul style={{ margin: 0, paddingLeft: '20px', fontSize: '13px' }}>
          <li><kbd>Ctrl/Cmd + F</kbd>: Search and filter</li>
          <li><kbd>Ctrl/Cmd + Z</kbd>: Undo surgery</li>
          <li><kbd>Ctrl/Cmd + Shift + Z</kbd>: Redo surgery</li>
        </ul>
      </div>
    </div>
  );
}

