import { describe, it, expect, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { loadShardedModel } from './shardedModel';
import { readTensorFromUnifiedMap } from './tensorReader';
import {
  createTestShardFile,
  createTestIndexFile,
  createFloat32Buffer,
  createTestSafetensorsFile,
  cleanupTestDir,
} from '../test/helpers/createTestSafetensors';

let testDirs: string[] = [];

function makeTempDir(): string {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'shard-test-'));
  testDirs.push(d);
  return d;
}

afterEach(() => {
  for (const d of testDirs) {
    cleanupTestDir(d);
  }
  testDirs = [];
});

describe('R103 -- Sharded Model Support', () => {
  it('loads a 3-shard model correctly', async () => {
    const dir = makeTempDir();

    const dataA = createFloat32Buffer(4);
    const dataB = createFloat32Buffer(8);
    const dataC = createFloat32Buffer(16);

    createTestShardFile(
      'model-00001-of-00003.safetensors',
      [{ name: 'layer.0.weight', dtype: 'F32', shape: [2, 2], data: dataA }],
      {},
      dir,
    );
    createTestShardFile(
      'model-00002-of-00003.safetensors',
      [{ name: 'layer.1.weight', dtype: 'F32', shape: [2, 4], data: dataB }],
      {},
      dir,
    );
    createTestShardFile(
      'model-00003-of-00003.safetensors',
      [{ name: 'layer.2.weight', dtype: 'F32', shape: [4, 4], data: dataC }],
      {},
      dir,
    );

    const indexPath = createTestIndexFile(
      {
        'layer.0.weight': 'model-00001-of-00003.safetensors',
        'layer.1.weight': 'model-00002-of-00003.safetensors',
        'layer.2.weight': 'model-00003-of-00003.safetensors',
      },
      dir,
    );

    const unified = await loadShardedModel(indexPath);

    expect(Object.keys(unified.tensors)).toHaveLength(3);
    expect(unified.tensors['layer.0.weight'].shardFile).toBe('model-00001-of-00003.safetensors');
    expect(unified.tensors['layer.1.weight'].shardFile).toBe('model-00002-of-00003.safetensors');
    expect(unified.tensors['layer.2.weight'].shardFile).toBe('model-00003-of-00003.safetensors');

    // Verify lazy reads route to correct shard
    const readA = await readTensorFromUnifiedMap(indexPath, unified, 'layer.0.weight');
    expect(readA.equals(dataA)).toBe(true);

    const readC = await readTensorFromUnifiedMap(indexPath, unified, 'layer.2.weight');
    expect(readC.equals(dataC)).toBe(true);
  });

  it('errors on missing shard file', async () => {
    const dir = makeTempDir();

    createTestShardFile(
      'model-00001-of-00002.safetensors',
      [{ name: 'w1', dtype: 'F32', shape: [2], data: createFloat32Buffer(2) }],
      {},
      dir,
    );

    // Index references a shard that doesn't exist on disk
    createTestIndexFile(
      {
        'w1': 'model-00001-of-00002.safetensors',
        'w2': 'model-00002-of-00002.safetensors', // missing
      },
      dir,
    );

    const indexPath = path.join(dir, 'model.safetensors.index.json');
    await expect(loadShardedModel(indexPath)).rejects.toThrow('Missing shard files');
  });

  it('handles a single-file model (no index)', async () => {
    const dir = makeTempDir();
    const data = createFloat32Buffer(4);
    const filePath = createTestSafetensorsFile(
      [{ name: 'tensor', dtype: 'F32', shape: [4], data }],
      {},
      dir,
    );

    const unified = await loadShardedModel(filePath);

    expect(Object.keys(unified.tensors)).toHaveLength(1);
    expect(unified.tensors['tensor']).toBeDefined();
    expect(unified.tensors['tensor'].shardFile).toBe('model.safetensors');

    const readData = await readTensorFromUnifiedMap(filePath, unified, 'tensor');
    expect(readData.equals(data)).toBe(true);
  });

  it('auto-detects sharded model when opening a shard file', async () => {
    const dir = makeTempDir();

    createTestShardFile(
      'model-00001-of-00002.safetensors',
      [{ name: 'w1', dtype: 'F32', shape: [2], data: createFloat32Buffer(2) }],
      {},
      dir,
    );
    createTestShardFile(
      'model-00002-of-00002.safetensors',
      [{ name: 'w2', dtype: 'F32', shape: [4], data: createFloat32Buffer(4) }],
      {},
      dir,
    );

    createTestIndexFile(
      {
        'w1': 'model-00001-of-00002.safetensors',
        'w2': 'model-00002-of-00002.safetensors',
      },
      dir,
    );

    // Open via a shard file, not the index
    const shardPath = path.join(dir, 'model-00001-of-00002.safetensors');
    const unified = await loadShardedModel(shardPath);

    expect(Object.keys(unified.tensors)).toHaveLength(2);
  });
});
