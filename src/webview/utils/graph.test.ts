import { describe, it, expect } from 'vitest';
import { buildGraph } from './graph';
import { ArchitectureNode } from '../../types/tree';

describe('Graph Builder', () => {
  const mockTree: ArchitectureNode = {
    name: 'root',
    fullPath: '',
    type: 'root',
    children: [
      {
        name: 'layer1',
        fullPath: 'layer1',
        type: 'block',
        children: [
          {
            name: 'attn',
            fullPath: 'layer1.attn',
            type: 'component',
            children: [
              {
                name: 'weight',
                fullPath: 'layer1.attn.weight',
                type: 'parameter',
                children: [],
                tensorInfo: { dtype: 'F16', shape: [1024, 1024] }
              }
            ]
          }
        ]
      }
    ]
  };

  it('builds nodes and edges for root when collapsed', () => {
    const loraMap = {};
    const expandedNodes = new Set<string>();
    
    const { nodes, edges } = buildGraph(mockTree, loraMap, expandedNodes, '', 'all', false, 'highlight');
    
    // With everything collapsed, we only see layer1
    expect(nodes.length).toBe(1);
    expect(nodes[0].id).toBe('A-layer1');
    expect(edges.length).toBe(0); // Only 1 node, so 0 top-level edges
  });

  it('builds nodes and edges when expanded', () => {
    const loraMap = {};
    const expandedNodes = new Set<string>(['layer1', 'layer1.attn']);
    
    const { nodes, edges } = buildGraph(mockTree, loraMap, expandedNodes, '', 'all', false, 'highlight');
    
    // layer1, layer1.attn, layer1.attn.weight
    expect(nodes.length).toBe(3);
    const ids = nodes.map(n => n.id);
    expect(ids).toContain('A-layer1');
    expect(ids).toContain('A-layer1.attn');
    expect(ids).toContain('A-layer1.attn.weight');
    
    // layer1.attn is inside layer1, layer1.attn.weight is inside layer1.attn
    // Our layout generates edges for siblings.
    // In this simple tree, there are no siblings, so 0 edges.
    expect(edges.length).toBe(0);
  });

  it('handles search queries (highlight)', () => {
    const loraMap = {};
    const expandedNodes = new Set<string>(['layer1', 'layer1.attn']);
    
    const { nodes } = buildGraph(mockTree, loraMap, expandedNodes, 'weight', 'all', false, 'highlight');
    
    const weightNode = nodes.find(n => n.id === 'A-layer1.attn.weight');
    expect(weightNode?.data.isHighlighted).toBe(true);
    expect(weightNode?.data.isDimmed).toBe(false);

    const layerNode = nodes.find(n => n.id === 'A-layer1');
    expect(layerNode?.data.isHighlighted).toBe(false);
    expect(layerNode?.data.isDimmed).toBe(true);
  });

  it('handles search queries (isolate)', () => {
    const loraMap = {};
    const expandedNodes = new Set<string>(['layer1', 'layer1.attn']);
    
    const { nodes } = buildGraph(mockTree, loraMap, expandedNodes, 'weight', 'all', false, 'isolate');
    
    // layer1 and layer1.attn do not match, but they contain 'weight' which matches,
    // so they are NOT hidden but might be dimmed.
    expect(nodes.length).toBe(3);
    const weightNode = nodes.find(n => n.id === 'A-layer1.attn.weight');
    expect(weightNode?.data.isHighlighted).toBe(true);
  });

  it('handles LoRA filtering', () => {
    const loraMap = { 'layer1.attn.weight': { r: 8, alpha: 16 } };
    const expandedNodes = new Set<string>(['layer1', 'layer1.attn']);
    
    // filterLora = true
    const { nodes } = buildGraph(mockTree, loraMap, expandedNodes, '', 'all', true, 'highlight');
    
    const layerNode = nodes.find(n => n.id === 'A-layer1');
    // layer1 has lora down the tree, so it should not be dimmed
    expect(layerNode?.data.hasLora).toBe(true);
    expect(layerNode?.data.isDimmed).toBe(false);
  });
});
