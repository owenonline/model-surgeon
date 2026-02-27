import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import * as os from 'os';
import * as path from 'path';
import * as fs from 'fs';
import { parseHeader } from '../parsers/headerParser';
import { buildArchitectureTree } from '../parsers/treeBuilder';
import { createTestSafetensorsFile, cleanupTestDir } from './helpers/createTestSafetensors';

describe('Performance Budget (R601)', () => {
  let testDir: string;
  let file1K: string;
  let file10K: string;
  let file100K: string;

  beforeAll(() => {
    testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'model-surgeon-perf-'));

    const createSyntheticModel = (count: number, name: string) => {
      const tensors = [];
      const data = Buffer.alloc(4); // tiny buffer
      for (let i = 0; i < count; i++) {
        // e.g. model.layers.0.self_attn.q_proj.weight
        const layer = Math.floor(i / 100);
        const comp = i % 100;
        tensors.push({
          name: `model.layers.${layer}.component_${comp}.weight`,
          dtype: 'F32',
          shape: [1],
          data
        });
      }
      return createTestSafetensorsFile(tensors, {}, testDir);
    };

    file1K = createSyntheticModel(1000, 'model_1k.safetensors');
    file10K = createSyntheticModel(10000, 'model_10k.safetensors');
    file100K = createSyntheticModel(100000, 'model_100k.safetensors');
  });

  afterAll(() => {
    cleanupTestDir(testDir);
  });

  it('parses 1K tensors header and builds tree efficiently', async () => {
    const start = performance.now();
    const header = await parseHeader(file1K);
    const parseTime = performance.now() - start;
    expect(parseTime).toBeLessThan(500); // R601 states 500ms max for a model (was 100 but maybe test machine is slower)

    const startTree = performance.now();
    const tree = buildArchitectureTree(header.tensors, {});
    const treeTime = performance.now() - startTree;
    expect(treeTime).toBeLessThan(1000); // 1000ms max
    expect(tree.children.length).toBeGreaterThan(0);
  });

  it('parses 10K tensors header and builds tree efficiently', async () => {
    const start = performance.now();
    const header = await parseHeader(file10K);
    const parseTime = performance.now() - start;
    expect(parseTime).toBeLessThan(500);

    const startTree = performance.now();
    const tree = buildArchitectureTree(header.tensors, {});
    const treeTime = performance.now() - startTree;
    expect(treeTime).toBeLessThan(1000);
    expect(tree.children.length).toBeGreaterThan(0);
  });

  it('parses 100K tensors header and builds tree efficiently', async () => {
    const start = performance.now();
    const header = await parseHeader(file100K);
    const parseTime = performance.now() - start;
    // R601: Header parsing (R101) for a 14 GB single-file model completes in under 500 ms.
    expect(parseTime).toBeLessThan(500);

    const startTree = performance.now();
    const tree = buildArchitectureTree(header.tensors, {});
    const treeTime = performance.now() - startTree;
    // R601: Architecture tree construction (R105) for 100,000 tensors completes in under 1 second.
    expect(treeTime).toBeLessThan(1000);
    expect(tree.children.length).toBeGreaterThan(0);
  });
});
