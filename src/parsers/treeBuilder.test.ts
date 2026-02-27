import { describe, it, expect } from 'vitest';
import { buildArchitectureTree } from './treeBuilder';
import { TensorInfo } from '../types/safetensors';
import { LoraAdapterMap } from '../types/lora';
import { ArchitectureNode } from '../types/tree';

function findNode(root: ArchitectureNode, path: string): ArchitectureNode | undefined {
  if (root.fullPath === path) return root;
  for (const child of root.children) {
    const found = findNode(child, path);
    if (found) return found;
  }
  return undefined;
}

describe('R105 -- Model Architecture Tree Builder', () => {
  it('builds a tree from LLaMA-style tensor names', () => {
    const tensors: Record<string, TensorInfo> = {
      'model.embed_tokens.weight': { dtype: 'F16', shape: [32000, 4096], dataOffsets: [0, 100] },
      'model.layers.0.self_attn.q_proj.weight': { dtype: 'F16', shape: [4096, 4096], dataOffsets: [100, 200] },
      'model.layers.0.self_attn.k_proj.weight': { dtype: 'F16', shape: [4096, 4096], dataOffsets: [200, 300] },
      'model.layers.0.self_attn.v_proj.weight': { dtype: 'F16', shape: [4096, 4096], dataOffsets: [300, 400] },
      'model.layers.0.mlp.gate_proj.weight': { dtype: 'F16', shape: [11008, 4096], dataOffsets: [400, 500] },
      'model.layers.1.self_attn.q_proj.weight': { dtype: 'F16', shape: [4096, 4096], dataOffsets: [500, 600] },
      'model.layers.1.mlp.gate_proj.weight': { dtype: 'F16', shape: [11008, 4096], dataOffsets: [600, 700] },
      'model.norm.weight': { dtype: 'F16', shape: [4096], dataOffsets: [700, 800] },
      'lm_head.weight': { dtype: 'F16', shape: [32000, 4096], dataOffsets: [800, 900] },
    };

    const tree = buildArchitectureTree(tensors);

    expect(tree.type).toBe('root');
    expect(tree.children.length).toBeGreaterThanOrEqual(2);

    // Layer 0 should exist
    const layer0 = findNode(tree, 'model.layers.0');
    expect(layer0).toBeDefined();
    expect(layer0!.type).toBe('block');
    expect(layer0!.blockIndex).toBe(0);

    // Layer 1 should exist
    const layer1 = findNode(tree, 'model.layers.1');
    expect(layer1).toBeDefined();
    expect(layer1!.blockIndex).toBe(1);

    // Check parameter leaf
    const qProj = findNode(tree, 'model.layers.0.self_attn.q_proj.weight');
    expect(qProj).toBeDefined();
    expect(qProj!.type).toBe('parameter');
    expect(qProj!.tensorInfo).toEqual({ dtype: 'F16', shape: [4096, 4096] });
  });

  it('builds a tree from GPT-style naming (h.0.attn)', () => {
    const tensors: Record<string, TensorInfo> = {
      'wte.weight': { dtype: 'F32', shape: [50257, 768], dataOffsets: [0, 100] },
      'wpe.weight': { dtype: 'F32', shape: [1024, 768], dataOffsets: [100, 200] },
      'h.0.attn.c_attn.weight': { dtype: 'F32', shape: [768, 2304], dataOffsets: [200, 300] },
      'h.0.attn.c_proj.weight': { dtype: 'F32', shape: [768, 768], dataOffsets: [300, 400] },
      'h.0.mlp.c_fc.weight': { dtype: 'F32', shape: [768, 3072], dataOffsets: [400, 500] },
      'h.1.attn.c_attn.weight': { dtype: 'F32', shape: [768, 2304], dataOffsets: [500, 600] },
      'ln_f.weight': { dtype: 'F32', shape: [768], dataOffsets: [600, 700] },
    };

    const tree = buildArchitectureTree(tensors);

    const h0 = findNode(tree, 'h.0');
    expect(h0).toBeDefined();
    expect(h0!.type).toBe('block');
    expect(h0!.blockIndex).toBe(0);

    const h1 = findNode(tree, 'h.1');
    expect(h1).toBeDefined();
    expect(h1!.blockIndex).toBe(1);
  });

  it('attaches LoRA adapters to parent component, not as tree children', () => {
    const tensors: Record<string, TensorInfo> = {
      'model.layers.0.self_attn.q_proj.weight': { dtype: 'F16', shape: [4096, 4096], dataOffsets: [0, 100] },
      'model.layers.0.self_attn.q_proj.lora_A.weight': { dtype: 'F16', shape: [8, 4096], dataOffsets: [100, 200] },
      'model.layers.0.self_attn.q_proj.lora_B.weight': { dtype: 'F16', shape: [4096, 8], dataOffsets: [200, 300] },
    };

    const loraMap: LoraAdapterMap = {
      'model.layers.0.self_attn.q_proj.weight': {
        baseTensorName: 'model.layers.0.self_attn.q_proj.weight',
        loraAName: 'model.layers.0.self_attn.q_proj.lora_A.weight',
        loraBName: 'model.layers.0.self_attn.q_proj.lora_B.weight',
        rank: 8,
        alpha: 16,
        aShape: [8, 4096],
        bShape: [4096, 8],
      },
    };

    const tree = buildArchitectureTree(tensors, loraMap);

    // LoRA tensors should NOT appear as tree nodes
    const loraANode = findNode(tree, 'model.layers.0.self_attn.q_proj.lora_A.weight');
    expect(loraANode).toBeUndefined();

    // The parent component should have adapters attached
    const qProjComponent = findNode(tree, 'model.layers.0.self_attn.q_proj');
    expect(qProjComponent).toBeDefined();
    expect(qProjComponent!.adapters).toBeDefined();
    expect(
      qProjComponent!.adapters!['model.layers.0.self_attn.q_proj.weight'],
    ).toBeDefined();
  });

  it('produces deterministic ordering: blocks by index, others alphabetically', () => {
    const tensors: Record<string, TensorInfo> = {
      'model.layers.2.weight': { dtype: 'F32', shape: [10], dataOffsets: [0, 40] },
      'model.layers.0.weight': { dtype: 'F32', shape: [10], dataOffsets: [40, 80] },
      'model.layers.1.weight': { dtype: 'F32', shape: [10], dataOffsets: [80, 120] },
      'model.norm.weight': { dtype: 'F32', shape: [10], dataOffsets: [120, 160] },
      'model.embed.weight': { dtype: 'F32', shape: [10], dataOffsets: [160, 200] },
    };

    const tree = buildArchitectureTree(tensors);
    const modelNode = findNode(tree, 'model');
    expect(modelNode).toBeDefined();

    const childNames = modelNode!.children.map((c) => c.name);
    // Numbered blocks first (sorted by index), then alpha
    expect(childNames).toEqual(['embed', 'layers', 'norm']);

    // Within 'layers', blocks should be sorted 0, 1, 2
    const layersNode = findNode(tree, 'model.layers');
    expect(layersNode).toBeDefined();
    const layerIndices = layersNode!.children.map((c) => c.blockIndex);
    expect(layerIndices).toEqual([0, 1, 2]);
  });

  it('handles models without LoRA (empty loraMap)', () => {
    const tensors: Record<string, TensorInfo> = {
      'weight': { dtype: 'F32', shape: [10], dataOffsets: [0, 40] },
      'bias': { dtype: 'F32', shape: [10], dataOffsets: [40, 80] },
    };

    const tree = buildArchitectureTree(tensors, {});
    expect(tree.children.length).toBe(2);
    expect(tree.children.every((c) => !c.adapters || Object.keys(c.adapters).length === 0)).toBe(true);
  });
});
