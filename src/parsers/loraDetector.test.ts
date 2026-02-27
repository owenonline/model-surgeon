import { describe, it, expect, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { detectLoraAdapters, parseAdapterConfig } from './loraDetector';
import { TensorInfo } from '../types/safetensors';
import {
  createTestAdapterConfig,
  cleanupTestDir,
} from '../test/helpers/createTestSafetensors';

let testDirs: string[] = [];

function makeTempDir(): string {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'lora-test-'));
  testDirs.push(d);
  return d;
}

afterEach(() => {
  for (const d of testDirs) {
    cleanupTestDir(d);
  }
  testDirs = [];
});

describe('R104 -- LoRA Adapter Detection', () => {
  it('identifies lora_A and lora_B pairs', () => {
    const tensors: Record<string, TensorInfo> = {
      'model.layers.0.self_attn.q_proj.weight': {
        dtype: 'F32',
        shape: [4096, 4096],
        dataOffsets: [0, 100],
      },
      'model.layers.0.self_attn.q_proj.lora_A.weight': {
        dtype: 'F32',
        shape: [8, 4096],
        dataOffsets: [100, 200],
      },
      'model.layers.0.self_attn.q_proj.lora_B.weight': {
        dtype: 'F32',
        shape: [4096, 8],
        dataOffsets: [200, 300],
      },
    };

    const loraMap = detectLoraAdapters(tensors);

    expect(Object.keys(loraMap)).toHaveLength(1);
    const baseName = 'model.layers.0.self_attn.q_proj.weight';
    expect(loraMap[baseName]).toBeDefined();
    expect(loraMap[baseName].loraAName).toBe(
      'model.layers.0.self_attn.q_proj.lora_A.weight',
    );
    expect(loraMap[baseName].loraBName).toBe(
      'model.layers.0.self_attn.q_proj.lora_B.weight',
    );
    expect(loraMap[baseName].rank).toBe(8);
    expect(loraMap[baseName].aShape).toEqual([8, 4096]);
    expect(loraMap[baseName].bShape).toEqual([4096, 8]);
  });

  it('strips PEFT prefix (base_model.model.) from base name', () => {
    const tensors: Record<string, TensorInfo> = {
      'base_model.model.layers.0.mlp.gate_proj.lora_A.weight': {
        dtype: 'F16',
        shape: [16, 4096],
        dataOffsets: [0, 100],
      },
      'base_model.model.layers.0.mlp.gate_proj.lora_B.weight': {
        dtype: 'F16',
        shape: [11008, 16],
        dataOffsets: [100, 200],
      },
    };

    const loraMap = detectLoraAdapters(tensors);
    const baseName = 'layers.0.mlp.gate_proj.weight';
    expect(loraMap[baseName]).toBeDefined();
    expect(loraMap[baseName].rank).toBe(16);
  });

  it('uses adapter_config.json values for rank and alpha', () => {
    const tensors: Record<string, TensorInfo> = {
      'model.q_proj.lora_A.weight': {
        dtype: 'F32',
        shape: [32, 4096],
        dataOffsets: [0, 100],
      },
      'model.q_proj.lora_B.weight': {
        dtype: 'F32',
        shape: [4096, 32],
        dataOffsets: [100, 200],
      },
    };

    const config = {
      r: 32,
      lora_alpha: 64,
      target_modules: ['q_proj'],
      lora_dropout: 0.1,
    };

    const loraMap = detectLoraAdapters(tensors, config);
    const pair = loraMap['model.q_proj.weight'];
    expect(pair.rank).toBe(32);
    expect(pair.alpha).toBe(64);
  });

  it('returns empty map when no LoRA adapters present', () => {
    const tensors: Record<string, TensorInfo> = {
      'model.weight': { dtype: 'F32', shape: [10], dataOffsets: [0, 40] },
      'model.bias': { dtype: 'F32', shape: [10], dataOffsets: [40, 80] },
    };

    const loraMap = detectLoraAdapters(tensors);
    expect(Object.keys(loraMap)).toHaveLength(0);
  });

  it('ignores orphaned lora_A without matching lora_B', () => {
    const tensors: Record<string, TensorInfo> = {
      'model.q.lora_A.weight': {
        dtype: 'F32',
        shape: [8, 64],
        dataOffsets: [0, 100],
      },
      // No matching lora_B
    };

    const loraMap = detectLoraAdapters(tensors);
    expect(Object.keys(loraMap)).toHaveLength(0);
  });

  it('parses adapter_config.json from disk', async () => {
    const dir = makeTempDir();
    createTestAdapterConfig(dir, {
      r: 16,
      lora_alpha: 32,
      target_modules: ['q_proj', 'v_proj', 'k_proj'],
      lora_dropout: 0.05,
    });

    // Create a dummy safetensors file in the same dir
    const dummyPath = path.join(dir, 'adapter_model.safetensors');
    fs.writeFileSync(dummyPath, Buffer.alloc(16));

    const config = await parseAdapterConfig(dummyPath);
    expect(config).not.toBeNull();
    expect(config!.r).toBe(16);
    expect(config!.lora_alpha).toBe(32);
    expect(config!.target_modules).toEqual(['q_proj', 'v_proj', 'k_proj']);
    expect(config!.lora_dropout).toBe(0.05);
  });

  it('returns null when no adapter_config.json exists', async () => {
    const dir = makeTempDir();
    const dummyPath = path.join(dir, 'model.safetensors');
    fs.writeFileSync(dummyPath, Buffer.alloc(16));

    const config = await parseAdapterConfig(dummyPath);
    expect(config).toBeNull();
  });
});
