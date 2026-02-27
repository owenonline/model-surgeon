import React from 'react';
import { Handle, Position } from 'reactflow';
import { ArchitectureNode } from '../../types/tree';

const typeColors: Record<string, string> = {
  block: 'var(--vscode-charts-blue, #4fc1ff)',
  component: 'var(--vscode-charts-purple, #c586c0)',
  parameter: 'var(--vscode-charts-foreground, #ccc)',
};

const dtypeBadgeColors: Record<string, string> = {
  BF16: '#e06c75',
  F16: '#e5c07b',
  F32: '#98c379',
  I8: '#56b6c2',
};

export function CustomNode({ data, selected }: { data: Record<string, unknown>; selected: boolean }) {
  const node = data.node as ArchitectureNode;
  const hasLora = data.hasLora as boolean;
  const isDimmed = data.isDimmed as boolean;
  const isHighlighted = data.isHighlighted as boolean;
  const isLeafOrCollapsed = data.isLeafOrCollapsed as boolean;
  const descendantCount = (data.descendantCount as number) ?? 0;
  const comparisonStatus = data.comparisonStatus as { status: string; shapeMismatch?: boolean } | undefined;
  const loraAdapters = (data.loraAdapters ?? []) as Array<{ adapterName: string }>;

  const loraTitle = loraAdapters.length > 0
    ? `LoRA: ${loraAdapters.map(a => a.adapterName).join(', ')}`
    : 'Has LoRA adapter';

  const accentColor = typeColors[node.type] ?? typeColors.component;

  let borderLeft = `3px solid ${accentColor}`;
  if (comparisonStatus) {
    if (comparisonStatus.status === 'onlyA' || comparisonStatus.status === 'onlyB') {
      borderLeft = '3px dashed var(--vscode-descriptionForeground, #888)';
    }
  }
  if (selected) {
    borderLeft = '3px solid var(--vscode-focusBorder, #007acc)';
  }

  const opacity = isDimmed ? 0.35 : 1;
  const outline = isHighlighted ? '2px solid var(--vscode-editor-findMatchHighlightBorder, #f0a30a)' : 'none';

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={`${node.type} ${node.name}${node.tensorInfo ? `, shape ${node.tensorInfo.shape.join(' by ')}, ${node.tensorInfo.dtype}` : ''}${hasLora ? ', has LoRA adapter' : ''}`}
      style={{
        width: '100%',
        height: '100%',
        borderLeft,
        borderRadius: '4px',
        padding: '6px 10px',
        backgroundColor: 'var(--vscode-editor-background)',
        color: 'var(--vscode-editor-foreground)',
        opacity,
        outline,
        boxSizing: 'border-box',
        position: 'relative',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        fontFamily: 'var(--vscode-font-family)',
        boxShadow: selected
          ? '0 0 0 1px var(--vscode-focusBorder, #007acc)'
          : '0 1px 3px rgba(0,0,0,0.3)',
        cursor: 'pointer',
        overflow: 'hidden',
      }}
    >
      <Handle type="target" position={Position.Top} style={{ background: accentColor, width: 6, height: 6, border: 'none' }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', minWidth: 0 }}>
        <span
          style={{
            fontWeight: 600,
            fontSize: '12px',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            flex: 1,
          }}
          title={node.fullPath}
        >
          {node.name}
        </span>

        {comparisonStatus?.shapeMismatch && (
          <span title="Shape mismatch" style={{ fontSize: '11px', flexShrink: 0 }}>
            &#x26A0;
          </span>
        )}

        {hasLora && (
          <span
            title={loraTitle}
            style={{
              width: '8px',
              height: '8px',
              backgroundColor: '#89d185',
              borderRadius: '50%',
              flexShrink: 0,
              boxShadow: '0 0 4px rgba(137,209,133,0.6)',
            }}
          />
        )}
      </div>

      <div
        style={{
          fontSize: '10px',
          color: 'var(--vscode-descriptionForeground)',
          marginTop: '2px',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
        }}
      >
        {node.type === 'parameter' && node.tensorInfo ? (
          <>
            <span style={{ fontFamily: 'monospace' }}>[{node.tensorInfo.shape.join(', ')}]</span>
            <span
              style={{
                fontSize: '9px',
                padding: '0 4px',
                borderRadius: '2px',
                backgroundColor: dtypeBadgeColors[node.tensorInfo.dtype] ?? 'var(--vscode-badge-background)',
                color: '#1e1e1e',
                fontWeight: 600,
              }}
            >
              {node.tensorInfo.dtype}
            </span>
          </>
        ) : isLeafOrCollapsed && descendantCount > 0 ? (
          <span>{descendantCount} tensors</span>
        ) : node.children.length > 0 ? (
          <span>{node.children.length} children</span>
        ) : null}
      </div>

      <Handle type="source" position={Position.Bottom} style={{ background: accentColor, width: 6, height: 6, border: 'none' }} />
    </div>
  );
}
