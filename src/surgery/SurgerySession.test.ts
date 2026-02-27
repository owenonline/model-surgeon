import { describe, it, expect, beforeEach } from 'vitest';
import { SurgerySession } from './SurgerySession';
import { UnifiedTensorMap, ShardedTensorInfo } from '../types/safetensors';

describe('SurgerySession', () => {
  let initialModel: UnifiedTensorMap;

  beforeEach(() => {
    initialModel = {
      metadata: { format: 'pt' },
      tensors: {
        'layers.0.self_attn.q_proj.weight': { dtype: 'F16', shape: [4096, 4096], dataOffsets: [0, 100], shardFile: 'model.safetensors' },
        'layers.0.self_attn.k_proj.weight': { dtype: 'F16', shape: [4096, 4096], dataOffsets: [100, 200], shardFile: 'model.safetensors' },
        'layers.0.self_attn.q_proj.lora_A.weight': { dtype: 'F16', shape: [32, 4096], dataOffsets: [200, 300], shardFile: 'model.safetensors' },
        'layers.0.self_attn.q_proj.lora_B.weight': { dtype: 'F16', shape: [4096, 32], dataOffsets: [300, 400], shardFile: 'model.safetensors' },
        'layers.1.self_attn.q_proj.weight': { dtype: 'F16', shape: [4096, 4096], dataOffsets: [400, 500], shardFile: 'model.safetensors' }
      },
      shardHeaderLengths: { 'model.safetensors': 1024 }
    };
  });

  it('R400: should maintain undo/redo stack', () => {
    const session = new SurgerySession(initialModel);
    expect(session.pendingChangesCount).toBe(0);
    
    session.removeTensor('layers.1.self_attn.q_proj.weight');
    expect(session.pendingChangesCount).toBe(1);
    expect(session.getCurrentState().tensors['layers.1.self_attn.q_proj.weight']).toBeUndefined();
    
    expect(session.undo()).toBe(true);
    expect(session.pendingChangesCount).toBe(0);
    expect(session.getCurrentState().tensors['layers.1.self_attn.q_proj.weight']).toBeDefined();
    
    expect(session.redo()).toBe(true);
    expect(session.pendingChangesCount).toBe(1);
    expect(session.getCurrentState().tensors['layers.1.self_attn.q_proj.weight']).toBeUndefined();
    
    expect(session.undo()).toBe(true);
    
    // Pushing a new state after undo clears redo stack
    session.renameComponent('layers.0.self_attn.k_proj.weight', 'k_proj_new.weight');
    expect(session.redo()).toBe(false);
  });

  it('R401: should rename component and its descendant tensors', () => {
    const session = new SurgerySession(initialModel);
    // targetPath is layers.0.self_attn, newName is attention
    session.renameComponent('layers.0.self_attn', 'attention');
    
    const state = session.getCurrentState();
    expect(state.tensors['layers.0.attention.q_proj.weight']).toBeDefined();
    expect(state.tensors['layers.0.attention.k_proj.weight']).toBeDefined();
    expect(state.tensors['layers.0.attention.q_proj.lora_A.weight']).toBeDefined();
    
    // Original ones should be gone
    expect(state.tensors['layers.0.self_attn.q_proj.weight']).toBeUndefined();
    expect(state.tensors['layers.1.self_attn.q_proj.weight']).toBeDefined(); // untouched
  });

  it('R402: should remove LoRA adapter tensors', () => {
    const session = new SurgerySession(initialModel);
    session.removeLoraAdapter('layers.0.self_attn.q_proj');
    
    const state = session.getCurrentState();
    expect(state.tensors['layers.0.self_attn.q_proj.weight']).toBeDefined(); // base weight remains
    expect(state.tensors['layers.0.self_attn.q_proj.lora_A.weight']).toBeUndefined();
    expect(state.tensors['layers.0.self_attn.q_proj.lora_B.weight']).toBeUndefined();
  });

  it('R403: should rename LoRA adapter prefix', () => {
    const session = new SurgerySession(initialModel);
    // Assuming adapter is targeted by its prefix if it's "layers.0"
    session.renameLoraAdapter('layers.0', 'base_model.layers.0');
    
    const state = session.getCurrentState();
    expect(state.tensors['base_model.layers.0.self_attn.q_proj.lora_A.weight']).toBeDefined();
    expect(state.tensors['layers.0.self_attn.q_proj.lora_A.weight']).toBeUndefined();
    // Base weight shouldn't be renamed by renameLoraAdapter
    expect(state.tensors['layers.0.self_attn.q_proj.weight']).toBeDefined();
  });

  it('R404: should replace component with source tensors', () => {
    const session = new SurgerySession(initialModel);
    const sourceTensors: Record<string, ShardedTensorInfo> = {
      'layers.1.self_attn.q_proj.weight': { dtype: 'F32', shape: [4096, 4096], dataOffsets: [0, 200], shardFile: 'modelB.safetensors' }
    };
    
    session.replaceComponent('layers.1.self_attn', sourceTensors, { 'modelB.safetensors': 2048 });
    
    const state = session.getCurrentState();
    // The replaced tensor should come from model B
    expect(state.tensors['layers.1.self_attn.q_proj.weight'].dtype).toBe('F32');
    expect(state.tensors['layers.1.self_attn.q_proj.weight'].shardFile).toBe('modelB.safetensors');
    expect(state.shardHeaderLengths['modelB.safetensors']).toBe(2048);
  });
});
