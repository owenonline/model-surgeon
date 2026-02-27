import React from 'react';
import { ArchitectureNode } from '../../types/tree';
import { LoraAdapterMap } from '../../types/lora';

interface DetailPanelProps {
  node: ArchitectureNode | null;
  loraMap: LoraAdapterMap;
  onClose: () => void;
  onLoadStats?: (node: ArchitectureNode) => void;
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function getTensorSize(shape: number[], dtype: string): number {
  let elements = 1;
  for (const dim of shape) {
    elements *= dim;
  }
  let bytesPerElement = 2; // default f16/bf16
  if (dtype === 'F32' || dtype === 'I32') bytesPerElement = 4;
  if (dtype === 'F64' || dtype === 'I64') bytesPerElement = 8;
  if (dtype === 'I8' || dtype === 'U8') bytesPerElement = 1;
  return elements * bytesPerElement;
}

export function DetailPanel({ node, loraMap, onClose, onLoadStats }: DetailPanelProps) {
  if (!node) return null;

  return (
    <div style={{
      width: '300px',
      borderLeft: '1px solid var(--vscode-panel-border)',
      backgroundColor: 'var(--vscode-sideBar-background)',
      padding: '16px',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      boxSizing: 'border-box',
      overflowY: 'auto'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
        <h3 style={{ margin: 0, fontSize: '14px', wordBreak: 'break-all' }}>Tensor Details</h3>
        <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--vscode-icon-foreground)', cursor: 'pointer' }}>✕</button>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <strong>Name:</strong>
        <div style={{ wordBreak: 'break-all', fontFamily: 'monospace', fontSize: '12px' }}>{node.fullPath}</div>
      </div>

      {node.tensorInfo ? (
        <>
          <div style={{ marginBottom: '8px' }}><strong>Dtype:</strong> {node.tensorInfo.dtype}</div>
          <div style={{ marginBottom: '8px' }}><strong>Shape:</strong> [{node.tensorInfo.shape.join(', ')}]</div>
          <div style={{ marginBottom: '16px' }}>
            <strong>Size:</strong> {formatBytes(getTensorSize(node.tensorInfo.shape, node.tensorInfo.dtype))}
          </div>

          <button 
            onClick={() => onLoadStats && onLoadStats(node)}
            style={{
              padding: '6px 12px',
              backgroundColor: 'var(--vscode-button-background)',
              color: 'var(--vscode-button-foreground)',
              border: 'none',
              cursor: 'pointer',
              marginBottom: '16px'
            }}
          >
            Load Statistics
          </button>
        </>
      ) : (
        <div style={{ marginBottom: '16px', fontStyle: 'italic' }}>Not a leaf tensor</div>
      )}

      {node.adapters && Object.keys(node.adapters).length > 0 && (
        <div>
          <h4 style={{ margin: '0 0 8px 0' }}>LoRA Adapters</h4>
          {Object.values(node.adapters).map((adapter, i) => (
            <div key={i} style={{ backgroundColor: 'var(--vscode-editor-background)', padding: '8px', borderRadius: '4px', marginBottom: '8px' }}>
              <div><strong>Rank:</strong> {adapter.r}</div>
              <div><strong>Alpha:</strong> {adapter.alpha}</div>
              <div><strong>Scale:</strong> {(adapter.alpha / adapter.r).toFixed(2)}</div>
            </div>
          ))}
        </div>
      )}
      
      {!node.adapters && loraMap[node.fullPath] && (
        <div>
          <h4 style={{ margin: '0 0 8px 0' }}>LoRA Adapter</h4>
          <div style={{ backgroundColor: 'var(--vscode-editor-background)', padding: '8px', borderRadius: '4px', marginBottom: '8px' }}>
            <div><strong>Rank:</strong> {loraMap[node.fullPath].r}</div>
            <div><strong>Alpha:</strong> {loraMap[node.fullPath].alpha}</div>
            <div><strong>Scale:</strong> {(loraMap[node.fullPath].alpha / loraMap[node.fullPath].r).toFixed(2)}</div>
          </div>
        </div>
      )}
    </div>
  );
}
