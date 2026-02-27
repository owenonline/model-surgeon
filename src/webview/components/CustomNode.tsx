import React from 'react';
import { Handle, Position } from 'reactflow';
import { ArchitectureNode } from '../../types/tree';

export function CustomNode({ data, selected }: any) {
  const node: ArchitectureNode = data.node;
  const isExpanded: boolean = data.isExpanded;
  const hasLora: boolean = data.hasLora;
  const isHighlighted: boolean = data.isHighlighted;
  const isDimmed: boolean = data.isDimmed;

  let borderColor = 'var(--vscode-editorBracketMatch-border1, #666)';
  let borderStyle = 'solid';
  if (selected) borderColor = 'var(--vscode-editorBracketMatch-border2, #007acc)';
  if (isHighlighted) borderColor = 'var(--vscode-editorWarning-foreground, #f0a30a)';
  if (hasLora && !data.comparisonStatus) borderColor = 'var(--vscode-testing-iconPassed, #89d185)';

  const comparisonStatus = data.comparisonStatus;
  if (comparisonStatus) {
    if (comparisonStatus.status === 'onlyA' || comparisonStatus.status === 'onlyB') {
      borderStyle = 'dashed';
      borderColor = 'var(--vscode-descriptionForeground, #888)';
    } else if (data.diffMetrics) {
      // gradient from green (sim ~ 1.0) to red (sim < 0.5)
      const sim = data.diffMetrics.cosineSimilarity;
      if (sim > 0.99) borderColor = 'var(--vscode-testing-iconPassed, #28a745)'; // green
      else if (sim > 0.8) borderColor = 'var(--vscode-testing-iconQueued, #ffc107)'; // yellow
      else borderColor = 'var(--vscode-testing-iconFailed, #dc3545)'; // red
    }
  }

  const isAbsent = comparisonStatus && (comparisonStatus.status === 'onlyA' || comparisonStatus.status === 'onlyB') && data.modelId !== comparisonStatus.status.replace('only', '');
  const opacity = isDimmed || isAbsent ? 0.3 : 1;

  if (!isExpanded || node.type === 'parameter' || node.children.length === 0) {
    return (
      <div
        role="button"
        tabIndex={0}
        aria-label={`${node.type} ${node.name}${node.tensorInfo ? `, shape ${node.tensorInfo.shape.join(' by ')}, ${node.tensorInfo.dtype}` : ''}${hasLora ? ', has LoRA adapter' : ''}`}
        style={{
          width: '100%',
          height: '100%',
          border: `2px ${borderStyle} ${borderColor}`,
          borderRadius: '4px',
          padding: '8px',
          backgroundColor: 'var(--vscode-editor-background)',
          color: 'var(--vscode-editor-foreground)',
          opacity,
          boxSizing: 'border-box',
          position: 'relative',
        }}
      >
        <Handle type="target" position={Position.Top} style={{ visibility: 'hidden' }} />
        <div style={{ fontWeight: 'bold', fontSize: '12px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', display: 'flex', justifyContent: 'space-between' }}>
          <span>{node.name}</span>
          {comparisonStatus?.shapeMismatch && <span title="Shape Mismatch">⚠️</span>}
        </div>
        {node.type === 'parameter' && node.tensorInfo && (
          <div style={{ fontSize: '10px', color: 'var(--vscode-descriptionForeground)' }}>
            [{node.tensorInfo.shape.join(', ')}] • {node.tensorInfo.dtype}
          </div>
        )}
        {node.type !== 'parameter' && (
          <div style={{ fontSize: '10px', color: 'var(--vscode-descriptionForeground)' }}>
            {node.children.length} children
          </div>
        )}
        {hasLora && (
          <div
            title="Has LoRA Adapter"
            style={{
              position: 'absolute',
              top: '-4px',
              right: '-4px',
              width: '12px',
              height: '12px',
              backgroundColor: '#89d185',
              borderRadius: '50%',
              border: '2px solid var(--vscode-editor-background)',
            }}
          />
        )}
        <Handle type="source" position={Position.Bottom} style={{ visibility: 'hidden' }} />
      </div>
    );
  }

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={`Expanded ${node.type} ${node.name}`}
      style={{
        width: '100%',
        height: '100%',
        border: `2px dashed ${borderColor}`,
        borderRadius: '8px',
        backgroundColor: 'rgba(128, 128, 128, 0.05)',
        opacity,
        boxSizing: 'border-box',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ visibility: 'hidden' }} />
      <div style={{ padding: '8px', fontWeight: 'bold', borderBottom: `1px dashed ${borderColor}`, marginBottom: '8px', fontSize: '14px', color: 'var(--vscode-editor-foreground)' }}>
        {node.name}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ visibility: 'hidden' }} />
    </div>
  );
}
