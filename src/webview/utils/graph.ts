import { Node, Edge, MarkerType } from 'reactflow';
import dagre from 'dagre';
import { ArchitectureNode } from '../../types/tree';
import { LoraAdapterMap } from '../../types/lora';
import { AlignedComponent } from '../../types/messages';

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

const NODE_WIDTH = 260;
const NODE_HEIGHT = 64;
const BLOCK_NODE_HEIGHT = 48;
const RANK_SEP = 60;
const NODE_SEP = 30;

function countDescendants(node: ArchitectureNode): number {
  let count = 0;
  for (const c of node.children) {
    count += 1 + countDescendants(c);
  }
  return count;
}

function collectVisibleNodes(
  node: ArchitectureNode,
  expandedNodes: Set<string>,
  loraMap: LoraAdapterMap,
  searchQuery: string,
  filterDtype: string,
  filterLora: boolean,
  filterMode: 'highlight' | 'isolate',
  modelId: 'A' | 'B',
  comparison: { alignedComponents: AlignedComponent[] } | undefined,
  result: { nodes: Array<{ id: string; data: Record<string, unknown>; width: number; height: number }>; edges: Array<{ source: string; target: string }> },
  parentPath: string | null,
): boolean {
  const isExpanded = expandedNodes.has(node.fullPath);
  const hasL = hasLoraDown(node, loraMap);

  const matchSearch = !searchQuery || node.fullPath.toLowerCase().includes(searchQuery.toLowerCase());
  const matchSearchDown = matchSearch || node.children.some((c) => matchesSearchDown(c, searchQuery));

  if (filterMode === 'isolate' && searchQuery && !matchSearch && !matchSearchDown) {
    return false;
  }

  let matchFilter = true;
  if (filterDtype && filterDtype !== 'all' && node.type === 'parameter' && node.tensorInfo?.dtype !== filterDtype) {
    matchFilter = false;
  }
  if (filterLora && !hasL) {
    matchFilter = false;
  }
  if (filterMode === 'isolate' && !matchFilter) {
    return false;
  }

  const isDimmed = (searchQuery && filterMode === 'highlight' && !matchSearch) || (!matchFilter && filterMode === 'highlight');

  let comparisonStatus: AlignedComponent | undefined;
  if (comparison) {
    const canonical = node.fullPath.replace(/^base_model\.model\./, '');
    comparisonStatus = comparison.alignedComponents.find((c) => c.path === canonical);
  }

  const isLeafOrCollapsed = !isExpanded || node.type === 'parameter' || node.children.length === 0;
  const descendantCount = countDescendants(node);

  const nodeId = `${modelId}-${node.fullPath}`;
  const height = node.type === 'parameter' ? NODE_HEIGHT : BLOCK_NODE_HEIGHT;

  result.nodes.push({
    id: nodeId,
    width: NODE_WIDTH,
    height,
    data: {
      label: node.name,
      node,
      modelId,
      isExpanded,
      hasLora: hasL,
      loraAdapters: node.adapters
        ? Object.values(node.adapters)
        : loraMap[node.fullPath] ?? [],
      isHighlighted: searchQuery ? matchSearch : false,
      isDimmed,
      comparisonStatus,
      descendantCount,
      isLeafOrCollapsed,
    },
  });

  if (parentPath !== null) {
    result.edges.push({
      source: `${modelId}-${parentPath}`,
      target: nodeId,
    });
  }

  if (!isLeafOrCollapsed) {
    for (const child of node.children) {
      collectVisibleNodes(child, expandedNodes, loraMap, searchQuery, filterDtype, filterLora, filterMode, modelId, comparison, result, node.fullPath);
    }
  }

  return true;
}

function hasLoraDown(node: ArchitectureNode, lMap: LoraAdapterMap): boolean {
  if (lMap[node.fullPath]?.length) return true;
  if (node.adapters && Object.keys(node.adapters).length > 0) return true;
  return node.children.some((c) => hasLoraDown(c, lMap));
}

function matchesSearchDown(node: ArchitectureNode, query: string): boolean {
  if (node.fullPath.toLowerCase().includes(query.toLowerCase())) return true;
  return node.children.some((c) => matchesSearchDown(c, query));
}

export function buildGraph(
  rootNode: ArchitectureNode,
  loraMap: LoraAdapterMap,
  expandedNodes: Set<string>,
  searchQuery: string,
  filterDtype: string,
  filterLora: boolean,
  filterMode: 'highlight' | 'isolate',
  comparison?: {
    treeB: ArchitectureNode;
    loraMapB: LoraAdapterMap;
    alignedComponents: AlignedComponent[];
  },
): GraphData {
  const collected = {
    nodes: [] as Array<{ id: string; data: Record<string, unknown>; width: number; height: number }>,
    edges: [] as Array<{ source: string; target: string }>,
  };

  for (const child of rootNode.children) {
    collectVisibleNodes(child, expandedNodes, loraMap, searchQuery, filterDtype, filterLora, filterMode, 'A', comparison, collected, null);
  }

  if (comparison) {
    for (const child of comparison.treeB.children) {
      collectVisibleNodes(child, expandedNodes, comparison.loraMapB, searchQuery, filterDtype, filterLora, filterMode, 'B', comparison, collected, null);
    }
  }

  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', ranksep: RANK_SEP, nodesep: NODE_SEP, marginx: 40, marginy: 40 });

  for (const n of collected.nodes) {
    g.setNode(n.id, { width: n.width, height: n.height });
  }
  for (const e of collected.edges) {
    g.setEdge(e.source, e.target);
  }

  dagre.layout(g);

  let offsetB = 0;
  if (comparison) {
    let maxXA = 0;
    let minXB = Infinity;
    for (const n of collected.nodes) {
      const pos = g.node(n.id);
      if (!pos) continue;
      const mid = n.id.startsWith('A-') ? 'A' : 'B';
      if (mid === 'A') maxXA = Math.max(maxXA, pos.x + n.width / 2);
      if (mid === 'B') minXB = Math.min(minXB, pos.x - n.width / 2);
    }
    if (minXB < maxXA + 300) {
      offsetB = maxXA + 300 - minXB;
    }
  }

  const nodes: Node[] = collected.nodes.map((n) => {
    const pos = g.node(n.id);
    const isB = n.id.startsWith('B-');
    return {
      id: n.id,
      type: 'customNode',
      position: {
        x: (pos?.x ?? 0) - n.width / 2 + (isB ? offsetB : 0),
        y: (pos?.y ?? 0) - n.height / 2,
      },
      data: n.data,
      style: { width: n.width, height: n.height },
    };
  });

  const edgeColor = 'var(--vscode-editorWidget-border, #555)';
  const edges: Edge[] = collected.edges.map((e, i) => ({
    id: `e-${i}`,
    source: e.source,
    target: e.target,
    type: 'smoothstep',
    animated: false,
    style: { stroke: edgeColor, strokeWidth: 1.5 },
    markerEnd: { type: MarkerType.ArrowClosed, width: 12, height: 12, color: edgeColor },
  }));

  if (comparison) {
    for (const align of comparison.alignedComponents) {
      if (align.status === 'matched') {
        const sourceId = `A-${align.path}`;
        const targetId = `B-${align.path}`;
        const hasSource = collected.nodes.some((n) => n.id === sourceId);
        const hasTarget = collected.nodes.some((n) => n.id === targetId);
        if (hasSource && hasTarget) {
          edges.push({
            id: `match-${align.path}`,
            source: sourceId,
            target: targetId,
            type: 'straight',
            style: { stroke: 'var(--vscode-descriptionForeground, #888)', strokeDasharray: '6 4', strokeWidth: 1 },
          });
        }
      }
    }
  }

  return { nodes, edges };
}
