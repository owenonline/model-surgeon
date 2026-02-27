import { Node, Edge } from 'reactflow';
import { ArchitectureNode } from '../../types/tree';
import { LoraAdapterMap } from '../../types/lora';
import { AlignedComponent } from '../../types/messages';

export interface GraphData {
  nodes: Node[];
  edges: Edge[];
}

const PADDING = 20;
const PARAM_HEIGHT = 50;
const PARAM_WIDTH = 250;
const HEADER_HEIGHT = 40;

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
  }
): GraphData {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  function hasLoraDown(node: ArchitectureNode, lMap: LoraAdapterMap): boolean {
    if (lMap[node.fullPath]) return true;
    if (node.adapters && Object.keys(node.adapters).length > 0) return true;
    return node.children.some(c => hasLoraDown(c, lMap));
  }

  function matchesSearch(node: ArchitectureNode): boolean {
    if (!searchQuery) return true;
    return node.fullPath.toLowerCase().includes(searchQuery.toLowerCase());
  }

  function matchesFilters(node: ArchitectureNode, lMap: LoraAdapterMap): boolean {
    let match = true;
    if (filterDtype && filterDtype !== 'all') {
      if (node.type === 'parameter' && node.tensorInfo?.dtype !== filterDtype) {
        match = false;
      }
    }
    if (filterLora && !hasLoraDown(node, lMap)) {
      match = false;
    }
    return match;
  }

  function matchesSearchDown(node: ArchitectureNode): boolean {
    if (matchesSearch(node)) return true;
    return node.children.some(matchesSearchDown);
  }

  function calculateLayout(
    node: ArchitectureNode,
    lMap: LoraAdapterMap,
    modelId: 'A' | 'B',
    parentId: string | null = null,
    offsetX: number = 0,
    offsetY: number = 0
  ): { width: number; height: number } {
    const isExpanded = expandedNodes.has(node.fullPath);
    const hasL = hasLoraDown(node, lMap);

    let isHighlighted = false;
    let isDimmed = false;

    if (searchQuery) {
      isHighlighted = matchesSearch(node);
      isDimmed = filterMode === 'highlight' && !isHighlighted;
      if (filterMode === 'isolate' && !isHighlighted && !matchesSearchDown(node)) {
        return { width: 0, height: 0 };
      }
    }

    if (!matchesFilters(node, lMap)) {
      isDimmed = true;
      if (filterMode === 'isolate') return { width: 0, height: 0 };
    }

    let comparisonStatus = undefined;
    if (comparison) {
      const canonical = node.fullPath.replace(/^base_model\.model\./, '');
      const aligned = comparison.alignedComponents.find(c => c.path === canonical);
      if (aligned) {
        comparisonStatus = aligned;
      }
    }

    const n: Node = {
      id: `${modelId}-${node.fullPath}`,
      type: 'customNode',
      position: { x: offsetX, y: offsetY },
      data: {
        label: node.name,
        node,
        modelId,
        isExpanded,
        hasLora: hasL,
        loraAdapters: node.adapters ? Object.values(node.adapters) : (lMap[node.fullPath] ? [lMap[node.fullPath]] : []),
        isHighlighted,
        isDimmed,
        comparisonStatus,
      },
      parentNode: parentId ? `${modelId}-${parentId}` : undefined,
      style: { width: PARAM_WIDTH, height: PARAM_HEIGHT },
    };

    if (!isExpanded || node.type === 'parameter' || node.children.length === 0) {
      // Collapsed or leaf node
      n.style = { width: PARAM_WIDTH, height: PARAM_HEIGHT };
      nodes.push(n);
      return { width: PARAM_WIDTH, height: PARAM_HEIGHT };
    }

    // Expanded node layout
    let currentY = HEADER_HEIGHT;
    let maxWidth = PARAM_WIDTH;

    for (let i = 0; i < node.children.length; i++) {
      const child = node.children[i];
      const size = calculateLayout(child, lMap, modelId, node.fullPath, PADDING, currentY);
      
      if (size.height > 0) {
        if (i > 0) {
          const prevChild = node.children[i - 1];
          edges.push({
            id: `e-${modelId}-${prevChild.fullPath}-${child.fullPath}`,
            source: `${modelId}-${prevChild.fullPath}`,
            target: `${modelId}-${child.fullPath}`,
            type: 'smoothstep',
          });
        }
        currentY += size.height + PADDING;
        maxWidth = Math.max(maxWidth, size.width + PADDING * 2);
      }
    }

    const finalWidth = maxWidth;
    const finalHeight = currentY;

    n.style = { width: finalWidth, height: finalHeight, backgroundColor: 'rgba(0,0,0,0.05)', border: '1px solid #ccc' };
    nodes.push(n);
    
    return { width: finalWidth, height: finalHeight };
  }

  let rootY_A = 0;
  let maxW_A = 0;
  for (const child of rootNode.children) {
    const size = calculateLayout(child, loraMap, 'A', null, 0, rootY_A);
    if (size.height > 0) {
      rootY_A += size.height + 50;
      maxW_A = Math.max(maxW_A, size.width);
    }
  }

  for (let i = 0; i < rootNode.children.length - 1; i++) {
    edges.push({
      id: `e-root-A-${i}`,
      source: `A-${rootNode.children[i].fullPath}`,
      target: `A-${rootNode.children[i+1].fullPath}`,
      type: 'smoothstep'
    });
  }

  if (comparison) {
    let rootY_B = 0;
    const offsetB = maxW_A + 500; // Place B to the right of A
    for (const child of comparison.treeB.children) {
      const size = calculateLayout(child, comparison.loraMapB, 'B', null, offsetB, rootY_B);
      if (size.height > 0) {
        rootY_B += size.height + 50;
      }
    }

    for (let i = 0; i < comparison.treeB.children.length - 1; i++) {
      edges.push({
        id: `e-root-B-${i}`,
        source: `B-${comparison.treeB.children[i].fullPath}`,
        target: `B-${comparison.treeB.children[i+1].fullPath}`,
        type: 'smoothstep'
      });
    }

    // Connect matched nodes between A and B
    for (const align of comparison.alignedComponents) {
      if (align.status === 'matched') {
        edges.push({
          id: `match-${align.path}`,
          source: `A-${align.path}`,
          target: `B-${align.path}`,
          type: 'straight',
          style: { stroke: '#888', strokeDasharray: '5 5' }
        });
      }
    }
  }

  return { nodes, edges };
}
