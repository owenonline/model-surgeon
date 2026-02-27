import { Node, Edge } from 'reactflow';
import { ArchitectureNode } from '../../types/tree';
import { LoraAdapterMap } from '../../types/lora';

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
  filterMode: 'highlight' | 'isolate'
): GraphData {
  const nodes: Node[] = [];
  const edges: Edge[] = [];

  function hasLoraDown(node: ArchitectureNode): boolean {
    if (loraMap[node.fullPath]) return true;
    if (node.adapters && Object.keys(node.adapters).length > 0) return true;
    return node.children.some(hasLoraDown);
  }

  function matchesSearch(node: ArchitectureNode): boolean {
    if (!searchQuery) return true;
    return node.fullPath.toLowerCase().includes(searchQuery.toLowerCase());
  }

  function matchesFilters(node: ArchitectureNode): boolean {
    let match = true;
    if (filterDtype && filterDtype !== 'all') {
      if (node.type === 'parameter' && node.tensorInfo?.dtype !== filterDtype) {
        match = false;
      }
    }
    if (filterLora && !hasLoraDown(node)) {
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
    parentId: string | null = null,
    offsetX: number = 0,
    offsetY: number = 0
  ): { width: number; height: number } {
    const isExpanded = expandedNodes.has(node.fullPath);
    const hasL = hasLoraDown(node);

    let isHighlighted = false;
    let isDimmed = false;

    if (searchQuery) {
      isHighlighted = matchesSearch(node);
      isDimmed = filterMode === 'highlight' && !isHighlighted;
      if (filterMode === 'isolate' && !isHighlighted && !matchesSearchDown(node)) {
        return { width: 0, height: 0 };
      }
    }

    if (!matchesFilters(node)) {
      isDimmed = true;
      if (filterMode === 'isolate') return { width: 0, height: 0 };
    }

    const n: Node = {
      id: node.fullPath,
      type: 'customNode',
      position: { x: offsetX, y: offsetY },
      data: {
        label: node.name,
        node,
        isExpanded,
        hasLora: hasL,
        loraAdapters: node.adapters ? Object.values(node.adapters) : (loraMap[node.fullPath] ? [loraMap[node.fullPath]] : []),
        isHighlighted,
        isDimmed,
      },
      parentNode: parentId || undefined,
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
      const size = calculateLayout(child, node.fullPath, PADDING, currentY);
      
      if (size.height > 0) {
        if (i > 0) {
          const prevChild = node.children[i - 1];
          edges.push({
            id: `e-${prevChild.fullPath}-${child.fullPath}`,
            source: prevChild.fullPath,
            target: child.fullPath,
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

  let rootY = 0;
  for (const child of rootNode.children) {
    const size = calculateLayout(child, null, 0, rootY);
    if (size.height > 0) {
      rootY += size.height + 50;
    }
  }

  // Generate top-level edges
  for (let i = 0; i < rootNode.children.length - 1; i++) {
    edges.push({
      id: `e-root-${i}`,
      source: rootNode.children[i].fullPath,
      target: rootNode.children[i+1].fullPath,
      type: 'smoothstep'
    });
  }

  return { nodes, edges };
}
