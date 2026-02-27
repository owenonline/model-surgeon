import { describe, it, expect, vi, beforeEach } from 'vitest';
import { computeDiff, ComputeDiffPayload } from './diffComputation';
import * as tensorReader from '../parsers/tensorReader';

vi.mock('../parsers/tensorReader', () => ({
  readTensorData: vi.fn(),
}));

describe('Weight Difference Computation (R302)', () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('should compute exact same tensors as zero difference and 1.0 cosine similarity', async () => {
    // F32 tensors, value = [1.0, 2.0, 3.0]
    const bufferA = Buffer.alloc(12);
    bufferA.writeFloatLE(1.0, 0);
    bufferA.writeFloatLE(2.0, 4);
    bufferA.writeFloatLE(3.0, 8);

    vi.mocked(tensorReader.readTensorData).mockResolvedValue(bufferA);

    const payload: ComputeDiffPayload = {
      pathA: 'A', headerLengthA: 0, offsetsA: [0, 12], dtypeA: 'F32',
      pathB: 'B', headerLengthB: 0, offsetsB: [0, 12], dtypeB: 'F32'
    };

    const result = await computeDiff(payload);
    expect(result.cosineSimilarity).toBeCloseTo(1.0, 5);
    expect(result.l2NormDiff).toBe(0);
    expect(result.maxAbsDiff).toBe(0);
    expect(result.meanAbsDiff).toBe(0);
  });

  it('should handle dtype conversion correctly (F32 vs F16)', async () => {
    // F32 tensor A: [1.0, -1.0]
    const bufferA = Buffer.alloc(8);
    bufferA.writeFloatLE(1.0, 0);
    bufferA.writeFloatLE(-1.0, 4);

    // F16 tensor B: [1.0, -1.0]
    // 1.0 in F16 = 0x3c00
    // -1.0 in F16 = 0xbc00
    const bufferB = Buffer.alloc(4);
    bufferB.writeUInt16LE(0x3c00, 0);
    bufferB.writeUInt16LE(0xbc00, 2);

    vi.mocked(tensorReader.readTensorData).mockImplementation(async (path) => {
      return path === 'A' ? bufferA : bufferB;
    });

    const payload: ComputeDiffPayload = {
      pathA: 'A', headerLengthA: 0, offsetsA: [0, 8], dtypeA: 'F32',
      pathB: 'B', headerLengthB: 0, offsetsB: [0, 4], dtypeB: 'F16'
    };

    const result = await computeDiff(payload);
    expect(result.cosineSimilarity).toBeCloseTo(1.0, 5);
    expect(result.l2NormDiff).toBe(0);
    expect(result.maxAbsDiff).toBe(0);
    expect(result.meanAbsDiff).toBe(0);
  });

  it('should compute actual differences correctly', async () => {
    // F32 tensor A: [1.0, 2.0]
    const bufferA = Buffer.alloc(8);
    bufferA.writeFloatLE(1.0, 0);
    bufferA.writeFloatLE(2.0, 4);

    // F32 tensor B: [3.0, 4.0]
    const bufferB = Buffer.alloc(8);
    bufferB.writeFloatLE(3.0, 0);
    bufferB.writeFloatLE(4.0, 4);

    vi.mocked(tensorReader.readTensorData).mockImplementation(async (path) => {
      return path === 'A' ? bufferA : bufferB;
    });

    const payload: ComputeDiffPayload = {
      pathA: 'A', headerLengthA: 0, offsetsA: [0, 8], dtypeA: 'F32',
      pathB: 'B', headerLengthB: 0, offsetsB: [0, 8], dtypeB: 'F32'
    };

    const result = await computeDiff(payload);
    
    // A = [1, 2], B = [3, 4]
    // dotProduct = 3 + 8 = 11
    // magA = sqrt(5), magB = sqrt(25) = 5
    // cosineSimilarity = 11 / (5 * sqrt(5)) = 11 / 11.18 = 0.9838...
    expect(result.cosineSimilarity).toBeCloseTo(0.9838, 3);
    
    // diff = [-2, -2]
    // l2NormDiff = sqrt(4 + 4) = sqrt(8) = 2.828...
    expect(result.l2NormDiff).toBeCloseTo(2.828, 3);
    
    // maxAbsDiff = 2
    expect(result.maxAbsDiff).toBe(2);
    
    // meanAbsDiff = 2
    expect(result.meanAbsDiff).toBe(2);
  });
});
