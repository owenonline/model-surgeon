import React, { useState } from 'react';
import { useMessage } from './hooks/useMessage';
import { ArchitectureNode } from '../types/tree';
import { LoraAdapterMap } from '../types/lora';

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

  return (
    <div style={{ padding: '16px', fontFamily: 'var(--vscode-font-family)' }}>
      {error && (
        <div style={{ color: 'var(--vscode-errorForeground)', marginBottom: '8px' }}>
          {error}
        </div>
      )}
      {loading && <div>Loading...</div>}
      {progress && (
        <div>
          {progress.label}: {Math.round(progress.percent)}%
        </div>
      )}
      {tree && (
        <div>
          <h2>Model: {tree.name}</h2>
          <p>Tensors with LoRA adapters: {Object.keys(loraMap).length}</p>
          <p>Top-level children: {tree.children.length}</p>
        </div>
      )}
      {!tree && !loading && !error && (
        <div>
          <h2>Model Surgeon</h2>
          <p>Open a safetensors file to get started.</p>
        </div>
      )}
    </div>
  );
}
