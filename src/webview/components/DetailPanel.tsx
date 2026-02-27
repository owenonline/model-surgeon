import React from 'react';
import { ArchitectureNode } from '../../types/tree';
import { LoraAdapterMap } from '../../types/lora';
import { AlignedComponent, TensorDiffMetrics } from '../../types/messages';

interface LivePreview {
  valuesA: number[];
  valuesB: number[];
  shape: number[];
  error?: string;
}

interface ModuleDiffEntry {
  path: string;
  metrics: TensorDiffMetrics | null;
  error?: string;
}

interface DetailPanelData {
  node: ArchitectureNode;
  modelId: 'A' | 'B';
  comparisonStatus?: AlignedComponent;
  diffMetrics: TensorDiffMetrics | null;
  livePreview?: LivePreview | null;
  isLoadingDiff?: boolean;
  loraAdapters: unknown[];
  moduleDiffs?: ModuleDiffEntry[] | null;
  isLoadingModuleDiff?: boolean;
  onRequestModuleDiff?: () => void;
  hasComparison?: boolean;
  leafTensorCount?: number;
}

interface DetailPanelProps {
  data: DetailPanelData;
  loraMap: LoraAdapterMap;
  comparison?: {
    treeB: ArchitectureNode;
    loraMapB: LoraAdapterMap;
    alignedComponents: AlignedComponent[];
  };
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
  for (const dim of shape) elements *= dim;
  let bytesPerElement = 2;
  if (dtype === 'F32' || dtype === 'I32') bytesPerElement = 4;
  if (dtype === 'F64' || dtype === 'I64') bytesPerElement = 8;
  if (dtype === 'I8' || dtype === 'U8') bytesPerElement = 1;
  return elements * bytesPerElement;
}

function cosineSimilarityColor(cs: number): string {
  if (cs > 0.999) return '#28a745';
  if (cs > 0.90) return '#ffc107';
  return '#dc3545';
}

function formatVal(v: number): string {
  if (!isFinite(v)) return String(v);
  if (Math.abs(v) === 0) return '0';
  if (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0)) {
    return v.toExponential(3);
  }
  return v.toFixed(4);
}

function ValuePreview({ preview }: { preview: LivePreview }) {
  if (preview.error) {
    return (
      <div style={{ color: '#dc3545', fontSize: 11, marginTop: 6 }}>
        Error reading tensor: {preview.error}
      </div>
    );
  }
  const n = preview.valuesA.length;
  if (n === 0) return null;

  const rows: React.ReactNode[] = [];
  for (let i = 0; i < n; i++) {
    const a = formatVal(preview.valuesA[i]);
    const b = preview.valuesB[i] !== undefined ? formatVal(preview.valuesB[i]) : null;
    const same = b !== null && Math.abs((preview.valuesA[i] ?? 0) - (preview.valuesB[i] ?? 0)) < 1e-7;
    rows.push(
      <tr key={i}>
        <td style={{ paddingRight: 8, color: '#aaa', fontSize: 9, textAlign: 'right' }}>{i}</td>
        <td style={{ fontFamily: 'monospace', fontSize: 10, paddingRight: 12, color: '#98c379' }}>{a}</td>
        {b !== null && (
          <td style={{ fontFamily: 'monospace', fontSize: 10, color: same ? '#6c757d' : '#e06c75' }}>{b}</td>
        )}
      </tr>,
    );
  }

  return (
    <div style={{ marginTop: 8 }}>
      <div style={{ fontSize: 10, color: 'var(--vscode-descriptionForeground)', marginBottom: 4 }}>
        First {n} values (flat){preview.valuesB.length > 0 ? ' — Model A / Model B' : ''}:
      </div>
      <div
        style={{
          backgroundColor: 'var(--vscode-editor-background)',
          borderRadius: 4,
          padding: '6px 8px',
          maxHeight: 200,
          overflowY: 'auto',
        }}
      >
        <table style={{ borderCollapse: 'collapse', width: '100%' }}>
          <thead>
            <tr>
              <th style={{ fontSize: 9, color: '#6c757d', textAlign: 'right', paddingRight: 8 }}>#</th>
              <th style={{ fontSize: 9, color: '#98c379', textAlign: 'left', paddingRight: 12 }}>
                {preview.valuesB.length > 0 ? 'Model A' : 'Value'}
              </th>
              {preview.valuesB.length > 0 && (
                <th style={{ fontSize: 9, color: '#e06c75', textAlign: 'left' }}>Model B</th>
              )}
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </div>
  );
}

function DiffMetricsPanel({ metrics }: { metrics: TensorDiffMetrics }) {
  const cs = metrics.cosineSimilarity;
  const barColor = cosineSimilarityColor(cs);
  return (
    <div
      style={{
        marginBottom: 16,
        padding: 8,
        backgroundColor: 'var(--vscode-editor-background)',
        borderRadius: 4,
        borderLeft: '3px solid ' + barColor,
      }}
    >
      <div style={{ fontWeight: 600, fontSize: 12, marginBottom: 6 }}>Diff Metrics</div>
      <div style={{ marginBottom: 6 }}>
        <div style={{ fontSize: 11, marginBottom: 2 }}>
          <strong>Cosine Similarity:</strong>{' '}
          <span style={{ color: barColor, fontWeight: 600 }}>{(cs * 100).toFixed(4)}%</span>
        </div>
        <div style={{ width: '100%', height: 4, backgroundColor: '#333', borderRadius: 2, overflow: 'hidden' }}>
          <div style={{ width: `${Math.max(0, cs * 100)}%`, height: '100%', backgroundColor: barColor, transition: 'width 0.3s' }} />
        </div>
        {cs <= 0.999 && (
          <div style={{ fontSize: 10, color: '#aaa', marginTop: 2 }}>
            Δ from identical: {((1 - cs) * 100).toFixed(4)}%
          </div>
        )}
      </div>
      <div style={{ fontSize: 11 }}><strong>L2 Norm Diff:</strong> {metrics.l2NormDiff.toExponential(3)}</div>
      <div style={{ fontSize: 11 }}><strong>Max |Δ|:</strong> {metrics.maxAbsDiff.toExponential(3)}</div>
      <div style={{ fontSize: 11 }}><strong>Mean |Δ|:</strong> {metrics.meanAbsDiff.toExponential(3)}</div>
    </div>
  );
}

function ModuleDiffTable({ entries }: { entries: ModuleDiffEntry[] }) {
  // Sort: errors last, then by ascending cosine similarity (most different first)
  const sorted = [...entries].sort((a, b) => {
    if (!a.metrics && !b.metrics) return 0;
    if (!a.metrics) return 1;
    if (!b.metrics) return -1;
    return a.metrics.cosineSimilarity - b.metrics.cosineSimilarity;
  });

  const changed = entries.filter((e) => e.metrics && e.metrics.cosineSimilarity < 0.999).length;
  const total = entries.length;

  return (
    <div>
      <div style={{ fontSize: 11, marginBottom: 6, color: 'var(--vscode-descriptionForeground)' }}>
        {changed} / {total} tensors changed
      </div>
      <div style={{ maxHeight: 320, overflowY: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 10 }}>
          <thead>
            <tr style={{ borderBottom: '1px solid var(--vscode-panel-border)' }}>
              <th style={{ textAlign: 'left', paddingBottom: 4, color: '#aaa', fontWeight: 400 }}>Tensor</th>
              <th style={{ textAlign: 'right', paddingBottom: 4, color: '#aaa', fontWeight: 400, paddingLeft: 8 }}>Cos Sim</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((entry) => {
              const name = entry.path.split('.').pop() ?? entry.path;
              const shortPath = entry.path.length > 48
                ? '…' + entry.path.slice(entry.path.length - 46)
                : entry.path;
              const cs = entry.metrics?.cosineSimilarity;
              const color = cs !== undefined ? cosineSimilarityColor(cs) : '#6c757d';
              return (
                <tr
                  key={entry.path}
                  title={entry.path}
                  style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
                >
                  <td
                    style={{
                      padding: '3px 0',
                      fontFamily: 'monospace',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      maxWidth: 160,
                      color: '#ddd',
                    }}
                    title={entry.path}
                  >
                    {entry.error ? (
                      <span style={{ color: '#dc3545' }}>{name} (error)</span>
                    ) : (
                      shortPath
                    )}
                  </td>
                  <td
                    style={{
                      textAlign: 'right',
                      paddingLeft: 8,
                      fontFamily: 'monospace',
                      color,
                      fontWeight: cs !== undefined && cs < 0.999 ? 600 : 400,
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {cs !== undefined ? (cs * 100).toFixed(3) + '%' : entry.error ? '—' : '?'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function DetailPanel({ data, loraMap: _loraMap, comparison: _comparison, onClose, onLoadStats }: DetailPanelProps) {
  if (!data || !data.node) return null;

  const node: ArchitectureNode = data.node;
  const comparisonStatus = data.comparisonStatus;
  const diffMetrics = data.diffMetrics;
  const modelId = data.modelId;
  const isParameter = node.type === 'parameter';

  return (
    <div
      style={{
        width: 320,
        borderLeft: '1px solid var(--vscode-panel-border)',
        backgroundColor: 'var(--vscode-sideBar-background)',
        padding: 16,
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        boxSizing: 'border-box',
        overflowY: 'auto',
        flexShrink: 0,
      }}
    >
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 12 }}>
        <h3 style={{ margin: 0, fontSize: 13, wordBreak: 'break-all', lineHeight: 1.3 }}>
          {isParameter ? 'Tensor' : 'Module'} Details{modelId ? ` (Model ${modelId})` : ''}
        </h3>
        <button
          onClick={onClose}
          style={{ background: 'none', border: 'none', color: 'var(--vscode-icon-foreground)', cursor: 'pointer', flexShrink: 0, paddingLeft: 8 }}
        >
          ✕
        </button>
      </div>

      {/* Full path */}
      <div style={{ marginBottom: 12 }}>
        <div style={{ fontSize: 10, color: 'var(--vscode-descriptionForeground)', marginBottom: 3, textTransform: 'uppercase', letterSpacing: '0.06em' }}>Path</div>
        <div style={{ wordBreak: 'break-all', fontFamily: 'monospace', fontSize: 11, color: 'var(--vscode-editor-foreground)' }}>
          {node.fullPath}
        </div>
      </div>

      {/* Comparison status badge */}
      {comparisonStatus && (
        <div
          style={{
            marginBottom: 12,
            padding: '6px 8px',
            backgroundColor: 'var(--vscode-editor-background)',
            borderRadius: 4,
            fontSize: 11,
          }}
        >
          <strong>Comparison:</strong>{' '}
          <span
            style={{
              color:
                comparisonStatus.status === 'matched' && !comparisonStatus.shapeMismatch
                  ? '#28a745'
                  : comparisonStatus.status === 'onlyA' || comparisonStatus.status === 'onlyB'
                  ? '#dc3545'
                  : '#ffc107',
            }}
          >
            {comparisonStatus.status}
          </span>
          {comparisonStatus.shapeMismatch && (
            <span style={{ color: '#ffc107', marginLeft: 8 }}>⚠ shape mismatch</span>
          )}
        </div>
      )}

      {/* Tensor-specific info */}
      {isParameter && node.tensorInfo ? (
        <>
          <div style={{ fontSize: 11, marginBottom: 4 }}>
            <strong>Dtype:</strong> {node.tensorInfo.dtype}
          </div>
          <div style={{ fontSize: 11, marginBottom: 4 }}>
            <strong>Shape:</strong>{' '}
            <span style={{ fontFamily: 'monospace' }}>[{node.tensorInfo.shape.join(', ')}]</span>
          </div>
          <div style={{ fontSize: 11, marginBottom: 12 }}>
            <strong>Size:</strong> {formatBytes(getTensorSize(node.tensorInfo.shape, node.tensorInfo.dtype))}
            {' '}
            <span style={{ color: '#6c757d' }}>
              ({node.tensorInfo.shape.reduce((a, b) => a * b, 1).toLocaleString()} elements)
            </span>
          </div>

          {/* Diff metrics or loading state */}
          {data.hasComparison && comparisonStatus?.status === 'matched' && !comparisonStatus?.shapeMismatch && (
            <>
              {data.isLoadingDiff && (
                <div style={{ marginBottom: 12, padding: 8, backgroundColor: 'var(--vscode-editor-background)', borderRadius: 4, fontSize: 11, color: '#6c757d' }}>
                  ⏳ Computing weight diff…
                </div>
              )}
              {!data.isLoadingDiff && diffMetrics && (
                <DiffMetricsPanel metrics={diffMetrics} />
              )}
              {!data.isLoadingDiff && !diffMetrics && (
                <div style={{ marginBottom: 12, padding: 8, backgroundColor: 'var(--vscode-editor-background)', borderRadius: 4, fontSize: 11, color: '#6c757d' }}>
                  Diff not yet computed.
                </div>
              )}
            </>
          )}

          {/* Value preview */}
          {data.livePreview && <ValuePreview preview={data.livePreview} />}

          {/* Load stats button */}
          <button
            onClick={() => onLoadStats && onLoadStats(node)}
            style={{
              marginTop: 12,
              padding: '5px 12px',
              backgroundColor: 'var(--vscode-button-background)',
              color: 'var(--vscode-button-foreground)',
              border: 'none',
              borderRadius: 3,
              cursor: 'pointer',
              fontSize: 11,
            }}
          >
            Load Statistics
          </button>
        </>
      ) : !isParameter ? (
        <>
          {/* Module: child count */}
          <div style={{ fontSize: 11, marginBottom: 12, color: 'var(--vscode-descriptionForeground)' }}>
            {node.children.length} direct children · {data.leafTensorCount ?? 0} leaf tensors
          </div>

          {/* Module diff section */}
          {data.hasComparison && (
            <div style={{ marginBottom: 12 }}>
              {data.isLoadingModuleDiff ? (
                <div style={{ fontSize: 11, color: '#6c757d', padding: 8, backgroundColor: 'var(--vscode-editor-background)', borderRadius: 4 }}>
                  ⏳ Computing diffs for {data.leafTensorCount} tensors…
                </div>
              ) : data.moduleDiffs ? (
                <ModuleDiffTable entries={data.moduleDiffs} />
              ) : (
                <button
                  onClick={data.onRequestModuleDiff}
                  style={{
                    padding: '5px 12px',
                    backgroundColor: 'var(--vscode-button-background)',
                    color: 'var(--vscode-button-foreground)',
                    border: 'none',
                    borderRadius: 3,
                    cursor: 'pointer',
                    fontSize: 11,
                    width: '100%',
                  }}
                >
                  Compare {data.leafTensorCount ?? 0} tensors in this module
                </button>
              )}
            </div>
          )}
        </>
      ) : (
        <div style={{ fontSize: 11, fontStyle: 'italic', color: 'var(--vscode-descriptionForeground)' }}>
          Not a leaf tensor
        </div>
      )}

      {/* LoRA adapters */}
      {data.loraAdapters && data.loraAdapters.length > 0 && (
        <div style={{ marginTop: 8 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>LoRA Adapters</div>
          {(data.loraAdapters as Array<Record<string, unknown>>).map((adapter, i) => (
            <div
              key={i}
              style={{
                backgroundColor: 'var(--vscode-editor-background)',
                padding: 8,
                borderRadius: 4,
                marginBottom: 8,
                fontSize: 11,
              }}
            >
              <div><strong>Adapter:</strong> {String(adapter.adapterName ?? '(default)')}</div>
              {adapter.rank != null && <div><strong>Rank:</strong> {String(adapter.rank)}</div>}
              {adapter.alpha != null && <div><strong>Alpha:</strong> {String(adapter.alpha)}</div>}
              {adapter.rank != null && adapter.alpha != null && (
                <div>
                  <strong>Scale:</strong>{' '}
                  {(Number(adapter.alpha) / Number(adapter.rank)).toFixed(3)}
                </div>
              )}
              {Array.isArray(adapter.aShape) && adapter.aShape.length > 0 && (
                <div><strong>A shape:</strong> [{(adapter.aShape as number[]).join(', ')}]</div>
              )}
              {Array.isArray(adapter.bShape) && adapter.bShape.length > 0 && (
                <div><strong>B shape:</strong> [{(adapter.bShape as number[]).join(', ')}]</div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
