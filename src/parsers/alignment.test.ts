import { describe, it, expect } from 'vitest';
import { alignArchitectures } from './alignment';
import { ArchitectureNode } from '../types/tree';

describe('Architecture Alignment (R301)', () => {
  it('should match identical components and parameters', () => {
    const treeA: ArchitectureNode = {
      name: 'root',
      fullPath: '',
      type: 'root',
      children: [
        {
          name: 'layers.0.self_attn.q_proj.weight',
          fullPath: 'layers.0.self_attn.q_proj.weight',
          type: 'parameter',
          children: [],
          tensorInfo: { dtype: 'F16', shape: [4096, 4096] },
        },
      ],
    };

    const treeB: ArchitectureNode = {
      name: 'root',
      fullPath: '',
      type: 'root',
      children: [
        {
          name: 'layers.0.self_attn.q_proj.weight',
          fullPath: 'layers.0.self_attn.q_proj.weight',
          type: 'parameter',
          children: [],
          tensorInfo: { dtype: 'F16', shape: [4096, 4096] },
        },
      ],
    };

    const alignment = alignArchitectures(treeA, treeB);
    expect(alignment).toHaveLength(1);
    expect(alignment[0]).toEqual({ path: 'layers.0.self_attn.q_proj.weight', status: 'matched' });
  });

  it('should flag shape mismatch for matched parameters', () => {
    const treeA: ArchitectureNode = {
      name: 'root',
      fullPath: '',
      type: 'root',
      children: [
        {
          name: 'layers.0.self_attn.q_proj.weight',
          fullPath: 'layers.0.self_attn.q_proj.weight',
          type: 'parameter',
          children: [],
          tensorInfo: { dtype: 'F16', shape: [4096, 4096] },
        },
      ],
    };

    const treeB: ArchitectureNode = {
      name: 'root',
      fullPath: '',
      type: 'root',
      children: [
        {
          name: 'layers.0.self_attn.q_proj.weight',
          fullPath: 'layers.0.self_attn.q_proj.weight',
          type: 'parameter',
          children: [],
          tensorInfo: { dtype: 'F16', shape: [4096, 2048] },
        },
      ],
    };

    const alignment = alignArchitectures(treeA, treeB);
    expect(alignment).toHaveLength(1);
    expect(alignment[0]).toEqual({ path: 'layers.0.self_attn.q_proj.weight', status: 'matched', shapeMismatch: true });
  });

  it('should handle onlyA and onlyB components', () => {
    const treeA: ArchitectureNode = {
      name: 'root',
      fullPath: '',
      type: 'root',
      children: [
        {
          name: 'layers.0.weight',
          fullPath: 'layers.0.weight',
          type: 'parameter',
          children: [],
          tensorInfo: { dtype: 'F16', shape: [100] },
        },
      ],
    };

    const treeB: ArchitectureNode = {
      name: 'root',
      fullPath: '',
      type: 'root',
      children: [
        {
          name: 'layers.1.weight',
          fullPath: 'layers.1.weight',
          type: 'parameter',
          children: [],
          tensorInfo: { dtype: 'F16', shape: [100] },
        },
      ],
    };

    const alignment = alignArchitectures(treeA, treeB);
    expect(alignment).toHaveLength(2);
    expect(alignment).toContainEqual({ path: 'layers.0.weight', status: 'onlyA' });
    expect(alignment).toContainEqual({ path: 'layers.1.weight', status: 'onlyB' });
  });

  it('should strip peft prefixes (base_model.model.)', () => {
    const treeA: ArchitectureNode = {
      name: 'root',
      fullPath: '',
      type: 'root',
      children: [
        {
          name: 'base_model.model.layers.0.weight',
          fullPath: 'base_model.model.layers.0.weight',
          type: 'parameter',
          children: [],
          tensorInfo: { dtype: 'F16', shape: [100] },
        },
      ],
    };

    const treeB: ArchitectureNode = {
      name: 'root',
      fullPath: '',
      type: 'root',
      children: [
        {
          name: 'layers.0.weight',
          fullPath: 'layers.0.weight',
          type: 'parameter',
          children: [],
          tensorInfo: { dtype: 'F16', shape: [100] },
        },
      ],
    };

    const alignment = alignArchitectures(treeA, treeB);
    expect(alignment).toHaveLength(1);
    expect(alignment[0]).toEqual({ path: 'layers.0.weight', status: 'matched' });
  });
});
