import React, { useState } from 'react';
import { useMessage } from './hooks/useMessage';
import { ArchitectureNode } from '../types/tree';
import { LoraAdapterMap } from '../types/lora';
import { ModelGraph } from './components/ModelGraph';

export function App() {
  const [tree, setTree] = useState<ArchitectureNode | null>(null);
  const [loraMap, setLoraMap] = useState<LoraAdapterMap>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState<{ label: string; percent: number } | null>(null);

  useMessage({
    modelLoaded: (msg) => {
      setTree(msg.tree);
      setLoraMap(msg.loraMap);
      setLoading(false);
      setError(null);
    },
    error: (msg) => {
      setError(msg.message);
      setLoading(false);
    },
    progress: (msg) => {
      setProgress({ label: msg.label, percent: msg.percent });
    },
    comparisonResult: () => {
      setLoading(false);
    },
    surgeryResult: (msg) => {
      if (msg.updatedTree) {
        setTree(msg.updatedTree);
      }
      if (msg.error) {
        setError(msg.error);
      }
      setLoading(false);
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
      <div style={{ padding: '16px' }}>
        {progress.label}: {Math.round(progress.percent)}%
      </div>
    );
  }

  if (tree) {
    return (
      <div style={{ width: '100vw', height: '100vh', margin: 0, padding: 0 }}>
        <ModelGraph tree={tree} loraMap={loraMap} onLoadStats={handleLoadStats} />
      </div>
    );
  }

  return (
    <div style={{ padding: '16px', fontFamily: 'var(--vscode-font-family)' }}>
      <h2>Model Surgeon</h2>
      <p>Open a safetensors file to get started.</p>
    </div>
  );
}

