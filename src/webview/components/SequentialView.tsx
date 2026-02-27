import React, {
  useState,
  useCallback,
  useRef,
  useMemo,
} from 'react';
import { ArchitectureNode } from '../../types/tree';
import { LoraAdapterMap } from '../../types/lora';
import { AlignedComponent, PROTOCOL_VERSION } from '../../types/messages';
import { Toolbar } from './Toolbar';
import { DetailPanel } from './DetailPanel';
import { ContextMenu, ContextMenuState } from './ContextMenu';
import { RenameModal } from './RenameModal';
import { postMessageToExtension } from '../hooks/useMessage';

export interface SequentialViewProps {
  tree: ArchitectureNode;
  loraMap: LoraAdapterMap;
  comparison?: {
    treeB: ArchitectureNode;
    loraMapB: LoraAdapterMap;
    alignedComponents: AlignedComponent[];
  };
  onLoadStats?: (node: ArchitectureNode) => void;
}

interface BlockCallbacks {
  onToggle: (path: string) => void;
  onSelect: (node: ArchitectureNode, modelId: 'A' | 'B') => void;
  onContextMenu: (e: React.MouseEvent, node: ArchitectureNode, modelId: 'A' | 'B') => void;
  onHoverPath: (path: string | null) => void;
}

function getComparisonColor(path: string, alignedComponents: AlignedComponent[] | undefined): string | null {
  if (!alignedComponents) return null;
  const canonical = path.replace(/^base_model\.model\./, '');
  const entry = alignedComponents.find((c) => c.path === canonical);
  if (!entry) return '#6c757d';

  if (entry.status === 'onlyA' || entry.status === 'onlyB') return '#dc3545'; // red — absent
  if (entry.shapeMismatch) return '#ffc107'; // yellow — structural mismatch

  // Use actual cosine similarity when available
  if (entry.diffMetrics) {
    const cs = entry.diffMetrics.cosineSimilarity;
    if (cs > 0.999) return '#28a745'; // green — effectively identical
    if (cs > 0.90) return '#ffc107';  // yellow — moderate drift
    return '#dc3545';                  // red — significant change (trained weights)
  }

  // Fallback for large tensors where diff was skipped
  return '#28a745';
}

function getAlignedStatus(path: string, alignedComponents: AlignedComponent[] | undefined): AlignedComponent | undefined {
  if (!alignedComponents) return undefined;
  const canonical = path.replace(/^base_model\.model\./, '');
  return alignedComponents.find((c) => c.path === canonical);
}

function countDescendantTensors(node: ArchitectureNode): number {
  if (node.type === 'parameter') return 1;
  let count = 0;
  for (const c of node.children) count += countDescendantTensors(c);
  return count;
}

function nodeMatchesSearch(node: ArchitectureNode, query: string): boolean {
  if (!query) return true;
  const q = query.toLowerCase();
  return node.fullPath.toLowerCase().includes(q) || node.name.toLowerCase().includes(q);
}

function subtreeMatchesSearch(node: ArchitectureNode, query: string): boolean {
  if (nodeMatchesSearch(node, query)) return true;
  return node.children.some((c) => subtreeMatchesSearch(c, query));
}

function hasLoraDown(node: ArchitectureNode, loraMap: LoraAdapterMap): boolean {
  if (loraMap[node.fullPath]?.length) return true;
  if (node.adapters && Object.keys(node.adapters).length > 0) return true;
  return node.children.some((c) => hasLoraDown(c, loraMap));
}

const TYPE_COLORS: Record<string, string> = {
  block: 'var(--vscode-charts-blue, #4fc1ff)',
  component: 'var(--vscode-charts-purple, #c586c0)',
  parameter: 'var(--vscode-charts-foreground, #aaaaaa)',
  root: 'transparent',
};

const DTYPE_COLORS: Record<string, string> = {
  BF16: '#e06c75',
  F16: '#e5c07b',
  F32: '#98c379',
  I8: '#56b6c2',
};

interface SequentialBlockProps {
  node: ArchitectureNode;
  depth: number;
  modelId: 'A' | 'B';
  expandedPaths: Set<string>;
  callbacks: BlockCallbacks;
  loraMap: LoraAdapterMap;
  searchQuery: string;
  filterDtype: string;
  filterLora: boolean;
  filterMode: 'highlight' | 'isolate';
  hoveredPath: string | null;
  alignedComponents?: AlignedComponent[];
  isMatchHighlighted?: boolean;
  onRegisterRef?: (path: string, el: HTMLDivElement | null) => void;
}

function SequentialBlock({
  node,
  depth,
  modelId,
  expandedPaths,
  callbacks,
  loraMap,
  searchQuery,
  filterDtype,
  filterLora,
  filterMode,
  hoveredPath,
  alignedComponents,
  isMatchHighlighted,
  onRegisterRef,
}: SequentialBlockProps) {
  const isExpanded = expandedPaths.has(node.fullPath);
  const isHovered = hoveredPath === node.fullPath;
  const hasLora = hasLoraDown(node, loraMap);
  const tensorCount = countDescendantTensors(node);

  const subtreeMatches = subtreeMatchesSearch(node, searchQuery);
  const matchesSearch = nodeMatchesSearch(node, searchQuery);
  const matchesLora = !filterLora || hasLora;
  const matchesDtype =
    filterDtype === 'all' || node.type !== 'parameter' || node.tensorInfo?.dtype === filterDtype;

  if (filterMode === 'isolate') {
    if (searchQuery && !subtreeMatches) return null;
    if (filterLora && !hasLora && node.children.every((c) => !hasLoraDown(c, loraMap))) return null;
    if (filterDtype !== 'all' && node.type === 'parameter' && node.tensorInfo?.dtype !== filterDtype) return null;
  }

  const isDimmed =
    filterMode === 'highlight' &&
    ((searchQuery && !matchesSearch && !subtreeMatches) ||
      (!matchesDtype && node.type === 'parameter') ||
      !matchesLora);

  const comparisonStatus = alignedComponents ? getAlignedStatus(node.fullPath, alignedComponents) : undefined;
  const comparisonColor =
    alignedComponents && (isHovered || isMatchHighlighted)
      ? getComparisonColor(node.fullPath, alignedComponents)
      : null;

  const typeColor = TYPE_COLORS[node.type] ?? TYPE_COLORS.component;
  const accentColor = comparisonColor ?? typeColor;
  const isTopLevel = depth === 0;

  const loraAdapters = node.adapters ? Object.values(node.adapters) : (loraMap[node.fullPath] ?? []);
  const loraTitle =
    loraAdapters.length > 0
      ? 'LoRA: ' + loraAdapters.map((a) => a.adapterName).join(', ')
      : 'Has LoRA adapter';

  const handleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (node.children.length > 0) callbacks.onToggle(node.fullPath);
      callbacks.onSelect(node, modelId);
    },
    [node, modelId, callbacks],
  );

  const handleContextMenu = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      callbacks.onContextMenu(e, node, modelId);
    },
    [node, modelId, callbacks],
  );

  const handleMouseEnter = useCallback(
    () => callbacks.onHoverPath(node.fullPath),
    [node.fullPath, callbacks],
  );
  const handleMouseLeave = useCallback(() => callbacks.onHoverPath(null), [callbacks]);

  const refCallback = useCallback(
    (el: HTMLDivElement | null) => onRegisterRef?.(node.fullPath, el),
    [node.fullPath, onRegisterRef],
  );

  return (
    <div ref={refCallback} style={{ opacity: isDimmed ? 0.4 : 1, marginLeft: depth === 0 ? 0 : 16 }}>
      <div
        role="button"
        tabIndex={0}
        onClick={handleClick}
        onContextMenu={handleContextMenu}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
        onKeyDown={(e) => e.key === 'Enter' && handleClick(e as unknown as React.MouseEvent)}
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: isTopLevel ? '8px 12px' : '5px 10px',
          borderLeft: '3px solid ' + accentColor,
          borderRadius: '0 4px 4px 0',
          backgroundColor:
            isHovered || isMatchHighlighted
              ? 'var(--vscode-list-hoverBackground, rgba(255,255,255,0.06))'
              : isTopLevel
              ? 'var(--vscode-editorGroupHeader-tabsBackground, rgba(255,255,255,0.03))'
              : 'transparent',
          cursor: node.children.length > 0 ? 'pointer' : 'default',
          userSelect: 'none',
          fontFamily: 'var(--vscode-font-family)',
          transition: 'background-color 0.1s, border-left-color 0.15s',
          marginBottom: isTopLevel ? 0 : 1,
          boxShadow: isTopLevel ? '0 1px 3px rgba(0,0,0,0.25)' : undefined,
        }}
        aria-expanded={node.children.length > 0 ? isExpanded : undefined}
        aria-label={node.type + ' ' + node.name}
      >
        <span
          style={{
            fontSize: 10,
            color: 'var(--vscode-descriptionForeground)',
            flexShrink: 0,
            width: 12,
            textAlign: 'center',
            visibility: node.children.length > 0 ? 'visible' : 'hidden',
          }}
        >
          {isExpanded ? '\u25bc' : '\u25b6'}
        </span>

        <span
          style={{
            fontSize: isTopLevel ? 13 : 12,
            fontWeight: isTopLevel ? 600 : 400,
            color: 'var(--vscode-editor-foreground)',
            flex: 1,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
          title={node.fullPath}
        >
          {node.name}
        </span>

        {node.type === 'parameter' && node.tensorInfo && (
          <span
            style={{
              fontSize: 10,
              color: 'var(--vscode-descriptionForeground)',
              fontFamily: 'monospace',
              flexShrink: 0,
            }}
          >
            {'[' + node.tensorInfo.shape.join(', ') + ']'}
            <span
              style={{
                marginLeft: 4,
                padding: '0 4px',
                borderRadius: 2,
                backgroundColor: DTYPE_COLORS[node.tensorInfo.dtype] ?? 'var(--vscode-badge-background)',
                color: '#1e1e1e',
                fontWeight: 600,
                fontSize: 9,
              }}
            >
              {node.tensorInfo.dtype}
            </span>
          </span>
        )}

        {node.type !== 'parameter' && node.children.length > 0 && (
          <span style={{ fontSize: 10, color: 'var(--vscode-descriptionForeground)', flexShrink: 0 }}>
            {isExpanded ? node.children.length + ' children' : tensorCount + ' tensors'}
          </span>
        )}

        {comparisonStatus?.shapeMismatch && (
          <span title="Shape mismatch between models" style={{ fontSize: 11, flexShrink: 0 }}>
            {'\u26a0\ufe0f'}
          </span>
        )}

        {hasLora && (
          <span
            title={loraTitle}
            style={{
              width: 8,
              height: 8,
              backgroundColor: '#89d185',
              borderRadius: '50%',
              flexShrink: 0,
              display: 'inline-block',
              boxShadow: '0 0 4px rgba(137,209,133,0.6)',
            }}
          />
        )}
      </div>

      {isExpanded && node.children.length > 0 && (
        <div
          style={{
            borderLeft: '1px solid ' + accentColor + '22',
            marginLeft: 3,
            marginTop: 1,
            marginBottom: 2,
          }}
        >
          {node.children.map((child) => (
            <SequentialBlock
              key={child.fullPath}
              node={child}
              depth={depth + 1}
              modelId={modelId}
              expandedPaths={expandedPaths}
              callbacks={callbacks}
              loraMap={loraMap}
              searchQuery={searchQuery}
              filterDtype={filterDtype}
              filterLora={filterLora}
              filterMode={filterMode}
              hoveredPath={hoveredPath}
              alignedComponents={alignedComponents}
              onRegisterRef={onRegisterRef}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function FlowArrow() {
  return (
    <div
      style={{
        paddingLeft: 24,
        paddingTop: 4,
        paddingBottom: 4,
        color: 'var(--vscode-descriptionForeground)',
        fontSize: 16,
        lineHeight: 1,
        userSelect: 'none',
        opacity: 0.5,
      }}
    >
      {'\u2193'}
    </div>
  );
}

interface ModelColumnProps {
  label: string;
  tree: ArchitectureNode;
  loraMap: LoraAdapterMap;
  modelId: 'A' | 'B';
  expandedPaths: Set<string>;
  callbacks: BlockCallbacks;
  searchQuery: string;
  filterDtype: string;
  filterLora: boolean;
  filterMode: 'highlight' | 'isolate';
  hoveredPath: string | null;
  matchHighlightedPath: string | null;
  alignedComponents?: AlignedComponent[];
  onRegisterRef?: (path: string, el: HTMLDivElement | null) => void;
}

function ModelColumn({
  label,
  tree,
  loraMap,
  modelId,
  expandedPaths,
  callbacks,
  searchQuery,
  filterDtype,
  filterLora,
  filterMode,
  hoveredPath,
  matchHighlightedPath,
  alignedComponents,
  onRegisterRef,
}: ModelColumnProps) {
  return (
    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <div
        style={{
          padding: '6px 12px',
          fontSize: 11,
          fontWeight: 600,
          color: 'var(--vscode-descriptionForeground)',
          borderBottom: '1px solid var(--vscode-panel-border)',
          letterSpacing: '0.05em',
          textTransform: 'uppercase',
          flexShrink: 0,
        }}
      >
        {label}
      </div>
      <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0 8px 12px' }}>
        {tree.children.map((child, i) => (
          <React.Fragment key={child.fullPath}>
            {i > 0 && <FlowArrow />}
            <SequentialBlock
              node={child}
              depth={0}
              modelId={modelId}
              expandedPaths={expandedPaths}
              callbacks={callbacks}
              loraMap={loraMap}
              searchQuery={searchQuery}
              filterDtype={filterDtype}
              filterLora={filterLora}
              filterMode={filterMode}
              hoveredPath={hoveredPath}
              alignedComponents={alignedComponents}
              isMatchHighlighted={matchHighlightedPath === child.fullPath}
              onRegisterRef={onRegisterRef}
            />
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

interface ComparisonLayoutProps {
  treeA: ArchitectureNode;
  treeB: ArchitectureNode;
  loraMapA: LoraAdapterMap;
  loraMapB: LoraAdapterMap;
  expandedPaths: Set<string>;
  callbacks: BlockCallbacks;
  searchQuery: string;
  filterDtype: string;
  filterLora: boolean;
  filterMode: 'highlight' | 'isolate';
  alignedComponents: AlignedComponent[];
}

function ComparisonLayout({
  treeA,
  treeB,
  loraMapA,
  loraMapB,
  expandedPaths,
  callbacks,
  searchQuery,
  filterDtype,
  filterLora,
  filterMode,
  alignedComponents,
}: ComparisonLayoutProps) {
  const [hoveredPath, setHoveredPath] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const refsA = useRef<Map<string, HTMLDivElement>>(new Map());
  const refsB = useRef<Map<string, HTMLDivElement>>(new Map());

  const handleRegisterRefA = useCallback((path: string, el: HTMLDivElement | null) => {
    if (el) refsA.current.set(path, el);
    else refsA.current.delete(path);
  }, []);

  const handleRegisterRefB = useCallback((path: string, el: HTMLDivElement | null) => {
    if (el) refsB.current.set(path, el);
    else refsB.current.delete(path);
  }, []);

  const wrappedCallbacks: BlockCallbacks = useMemo(
    () => ({
      ...callbacks,
      onHoverPath: (path) => {
        setHoveredPath(path);
        callbacks.onHoverPath(path);
      },
    }),
    [callbacks],
  );

  const connectorLine = useMemo<{
    y1: number;
    y2: number;
    x1: number;
    x2: number;
    color: string;
    dashed: boolean;
  } | null>(() => {
    if (!hoveredPath || !containerRef.current) return null;
    const elA = refsA.current.get(hoveredPath);
    const elB = refsB.current.get(hoveredPath);
    if (!elA || !elB) return null;
    const containerRect = containerRef.current.getBoundingClientRect();
    const rectA = elA.getBoundingClientRect();
    const rectB = elB.getBoundingClientRect();
    const y1 = rectA.top - containerRect.top + rectA.height / 2;
    const y2 = rectB.top - containerRect.top + rectB.height / 2;
    const x1 = rectA.right - containerRect.left;
    const x2 = rectB.left - containerRect.left;
    const canonical = hoveredPath.replace(/^base_model\.model\./, '');
    const alignEntry = alignedComponents.find((c) => c.path === canonical);
    let color = '#6c757d';
    let dashed = true;
    if (alignEntry) {
      if (alignEntry.status === 'matched' && !alignEntry.shapeMismatch) {
        color = '#28a745';
        dashed = false;
      } else if (alignEntry.status === 'matched' && alignEntry.shapeMismatch) {
        color = '#ffc107';
        dashed = false;
      } else {
        color = '#dc3545';
        dashed = true;
      }
    }
    return { y1, y2, x1, x2, color, dashed };
  }, [hoveredPath, alignedComponents]);

  return (
    <div ref={containerRef} style={{ display: 'flex', flex: 1, overflow: 'hidden', position: 'relative' }}>
      <ModelColumn
        label="Model A"
        tree={treeA}
        loraMap={loraMapA}
        modelId="A"
        expandedPaths={expandedPaths}
        callbacks={wrappedCallbacks}
        searchQuery={searchQuery}
        filterDtype={filterDtype}
        filterLora={filterLora}
        filterMode={filterMode}
        hoveredPath={hoveredPath}
        matchHighlightedPath={hoveredPath}
        alignedComponents={alignedComponents}
        onRegisterRef={handleRegisterRefA}
      />
      <div style={{ width: 1, flexShrink: 0, backgroundColor: 'var(--vscode-panel-border)' }} />
      <ModelColumn
        label="Model B"
        tree={treeB}
        loraMap={loraMapB}
        modelId="B"
        expandedPaths={expandedPaths}
        callbacks={wrappedCallbacks}
        searchQuery={searchQuery}
        filterDtype={filterDtype}
        filterLora={filterLora}
        filterMode={filterMode}
        hoveredPath={hoveredPath}
        matchHighlightedPath={hoveredPath}
        alignedComponents={alignedComponents}
        onRegisterRef={handleRegisterRefB}
      />
      {connectorLine && (
        <svg
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: 'none',
            overflow: 'visible',
          }}
        >
          <line
            x1={connectorLine.x1}
            y1={connectorLine.y1}
            x2={connectorLine.x2}
            y2={connectorLine.y2}
            stroke={connectorLine.color}
            strokeWidth={2}
            strokeDasharray={connectorLine.dashed ? '6 4' : undefined}
            opacity={0.85}
          />
          <circle cx={connectorLine.x1} cy={connectorLine.y1} r={3} fill={connectorLine.color} />
          <circle cx={connectorLine.x2} cy={connectorLine.y2} r={3} fill={connectorLine.color} />
        </svg>
      )}
    </div>
  );
}

export function SequentialView({ tree, loraMap, comparison, onLoadStats }: SequentialViewProps) {
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [filterMode, setFilterMode] = useState<'highlight' | 'isolate'>('highlight');
  const [filterDtype, setFilterDtype] = useState('all');
  const [filterLora, setFilterLora] = useState(false);
  const [selectedNodeData, setSelectedNodeData] = useState<{
    node: ArchitectureNode;
    modelId: 'A' | 'B';
    comparisonStatus?: AlignedComponent;
    diffMetrics?: import('../../types/messages').TensorDiffMetrics;
    loraAdapters: unknown[];
  } | null>(null);
  const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
  const [renameTarget, setRenameTarget] = useState<ArchitectureNode | null>(null);

  const handleExpandAll = useCallback(() => {
    const all = new Set<string>();
    function traverse(n: ArchitectureNode) {
      if (n.children.length > 0) {
        all.add(n.fullPath);
        n.children.forEach(traverse);
      }
    }
    traverse(tree);
    if (comparison) traverse(comparison.treeB);
    setExpandedPaths(all);
  }, [tree, comparison]);

  const handleCollapseAll = useCallback(() => setExpandedPaths(new Set()), []);

  const handleToggle = useCallback((path: string) => {
    setExpandedPaths((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  }, []);

  const handleSelect = useCallback(
    (node: ArchitectureNode, modelId: 'A' | 'B') => {
      const aligned = comparison?.alignedComponents;
      const canonical = node.fullPath.replace(/^base_model\.model\./, '');
      const comparisonStatus = aligned?.find((c) => c.path === canonical);
      const currentLoraMap = modelId === 'B' ? comparison?.loraMapB ?? {} : loraMap;
      const loraAdapters = node.adapters
        ? Object.values(node.adapters)
        : currentLoraMap[node.fullPath] ?? [];
      setSelectedNodeData({ node, modelId, comparisonStatus, diffMetrics: comparisonStatus?.diffMetrics, loraAdapters });
    },
    [loraMap, comparison],
  );

  const handleContextMenuOpen = useCallback(
    (e: React.MouseEvent, node: ArchitectureNode, modelId: 'A' | 'B') => {
      const currentLoraMap = modelId === 'B' ? comparison?.loraMapB ?? {} : loraMap;
      const hasLora =
        (node.adapters && Object.keys(node.adapters).length > 0) ||
        (currentLoraMap[node.fullPath]?.length ?? 0) > 0;
      setContextMenu({ x: e.clientX, y: e.clientY, node, modelId, hasLora, hasComparison: !!comparison });
    },
    [loraMap, comparison],
  );

  const handleHoverPath = useCallback((_path: string | null) => {}, []);

  const callbacks: BlockCallbacks = useMemo(
    () => ({
      onToggle: handleToggle,
      onSelect: handleSelect,
      onContextMenu: handleContextMenuOpen,
      onHoverPath: handleHoverPath,
    }),
    [handleToggle, handleSelect, handleContextMenuOpen, handleHoverPath],
  );

  const handleRename = useCallback((node: ArchitectureNode) => setRenameTarget(node), []);
  const handleRenameConfirm = useCallback(
    (newName: string) => {
      if (!renameTarget) return;
      postMessageToExtension({
        type: 'performSurgery',
        protocolVersion: PROTOCOL_VERSION,
        operation: { operationType: 'renameTensor', targetPath: renameTarget.fullPath, newName },
      });
      setRenameTarget(null);
    },
    [renameTarget],
  );
  const handleRemove = useCallback((node: ArchitectureNode) => {
    postMessageToExtension({
      type: 'performSurgery',
      protocolVersion: PROTOCOL_VERSION,
      operation: { operationType: 'removeTensor', targetPath: node.fullPath },
    });
  }, []);
  const handleRemoveLora = useCallback((node: ArchitectureNode) => {
    postMessageToExtension({
      type: 'performSurgery',
      protocolVersion: PROTOCOL_VERSION,
      operation: { operationType: 'removeLoraAdapter', targetPath: node.fullPath },
    });
  }, []);
  const handleReplaceFromB = useCallback((node: ArchitectureNode) => {
    postMessageToExtension({
      type: 'performSurgery',
      protocolVersion: PROTOCOL_VERSION,
      operation: { operationType: 'replaceTensor', targetPath: node.fullPath, sourceModel: 'B' },
    });
  }, []);

  const detailPanelData = useMemo(() => {
    if (!selectedNodeData) return null;
    return {
      node: selectedNodeData.node,
      modelId: selectedNodeData.modelId,
      comparisonStatus: selectedNodeData.comparisonStatus,
      diffMetrics: selectedNodeData.diffMetrics,
      loraAdapters: selectedNodeData.loraAdapters,
    };
  }, [selectedNodeData]);

  return (
    <div
      style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw', overflow: 'hidden' }}
      onClick={() => contextMenu && setContextMenu(null)}
    >
      <Toolbar
        onExpandAll={handleExpandAll}
        onCollapseAll={handleCollapseAll}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        filterMode={filterMode}
        setFilterMode={setFilterMode}
        filterDtype={filterDtype}
        setFilterDtype={setFilterDtype}
        filterLora={filterLora}
        setFilterLora={setFilterLora}
        onFitView={handleCollapseAll}
      />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
          {comparison ? (
            <ComparisonLayout
              treeA={tree}
              treeB={comparison.treeB}
              loraMapA={loraMap}
              loraMapB={comparison.loraMapB}
              expandedPaths={expandedPaths}
              callbacks={callbacks}
              searchQuery={searchQuery}
              filterDtype={filterDtype}
              filterLora={filterLora}
              filterMode={filterMode}
              alignedComponents={comparison.alignedComponents}
            />
          ) : (
            <div style={{ flex: 1, overflowY: 'auto', padding: '8px 0 8px 12px' }}>
              {tree.children.map((child, i) => (
                <React.Fragment key={child.fullPath}>
                  {i > 0 && <FlowArrow />}
                  <SequentialBlock
                    node={child}
                    depth={0}
                    modelId="A"
                    expandedPaths={expandedPaths}
                    callbacks={callbacks}
                    loraMap={loraMap}
                    searchQuery={searchQuery}
                    filterDtype={filterDtype}
                    filterLora={filterLora}
                    filterMode={filterMode}
                    hoveredPath={null}
                  />
                </React.Fragment>
              ))}
            </div>
          )}
        </div>
        {detailPanelData && (
          <DetailPanel
            data={detailPanelData}
            loraMap={selectedNodeData?.modelId === 'B' ? comparison?.loraMapB ?? {} : loraMap}
            comparison={comparison}
            onClose={() => setSelectedNodeData(null)}
            onLoadStats={(n) => onLoadStats?.(n)}
          />
        )}
      </div>
      {contextMenu && (
        <ContextMenu
          {...contextMenu}
          onClose={() => setContextMenu(null)}
          onRename={handleRename}
          onRemove={handleRemove}
          onRemoveLora={handleRemoveLora}
          onReplaceFromB={handleReplaceFromB}
        />
      )}
      {renameTarget && (
        <RenameModal
          node={renameTarget}
          onConfirm={handleRenameConfirm}
          onCancel={() => setRenameTarget(null)}
        />
      )}
    </div>
  );
}
