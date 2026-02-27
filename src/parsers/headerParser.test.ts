import { describe, it, expect, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { parseHeader } from './headerParser';
import {
  createTestSafetensorsFile,
  createFloat32Buffer,
  cleanupTestDir,
} from '../test/helpers/createTestSafetensors';

let testDirs: string[] = [];

function makeTempDir(): string {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'parser-test-'));
  testDirs.push(d);
  return d;
}

afterEach(() => {
  for (const d of testDirs) {
    cleanupTestDir(d);
  }
  testDirs = [];
});

describe('R101 -- Safetensors Header Parser', () => {
  it('parses a valid single-file model', async () => {
    const dir = makeTempDir();
    const filePath = createTestSafetensorsFile(
      [
        { name: 'weight_a', dtype: 'F32', shape: [4, 4], data: createFloat32Buffer(16) },
        { name: 'weight_b', dtype: 'BF16', shape: [2, 3], data: Buffer.alloc(2 * 3 * 2) },
      ],
      { format: 'pt' },
      dir,
    );

    const result = await parseHeader(filePath);

    expect(result.metadata).toEqual({ format: 'pt' });
    expect(Object.keys(result.tensors)).toHaveLength(2);

    expect(result.tensors['weight_a']).toEqual({
      dtype: 'F32',
      shape: [4, 4],
      dataOffsets: expect.any(Array),
    });
    expect(result.tensors['weight_a'].dataOffsets[1] - result.tensors['weight_a'].dataOffsets[0]).toBe(64);

    expect(result.tensors['weight_b']).toEqual({
      dtype: 'BF16',
      shape: [2, 3],
      dataOffsets: expect.any(Array),
    });
    expect(result.tensors['weight_b'].dataOffsets[1] - result.tensors['weight_b'].dataOffsets[0]).toBe(12);

    expect(result.headerLength).toBeGreaterThan(0);
  });

  it('handles all supported dtypes', async () => {
    const dir = makeTempDir();
    const dtypes = ['F64', 'F32', 'F16', 'BF16', 'I64', 'I32', 'I16', 'I8', 'U8', 'BOOL', 'F8_E4M3', 'F8_E5M2'];
    const byteSizes: Record<string, number> = {
      F64: 8, F32: 4, F16: 2, BF16: 2, I64: 8, I32: 4, I16: 2, I8: 1, U8: 1, BOOL: 1, F8_E4M3: 1, F8_E5M2: 1,
    };

    const tensors = dtypes.map((dtype) => ({
      name: `tensor_${dtype}`,
      dtype,
      shape: [2],
      data: Buffer.alloc(2 * byteSizes[dtype]),
    }));

    const filePath = createTestSafetensorsFile(tensors, {}, dir);
    const result = await parseHeader(filePath);

    for (const dtype of dtypes) {
      expect(result.tensors[`tensor_${dtype}`]).toBeDefined();
      expect(result.tensors[`tensor_${dtype}`].dtype).toBe(dtype);
    }
  });

  it('rejects an invalid header (corrupt JSON)', async () => {
    const dir = makeTempDir();
    const filePath = path.join(dir, 'bad.safetensors');

    // Write a file with invalid JSON as the header
    const headerBuf = Buffer.from('{ this is not valid json !!!');
    const lengthBuf = Buffer.alloc(8);
    lengthBuf.writeBigUInt64LE(BigInt(headerBuf.length), 0);

    fs.writeFileSync(filePath, Buffer.concat([lengthBuf, headerBuf]));

    await expect(parseHeader(filePath)).rejects.toThrow('not valid JSON');
  });

  it('rejects an oversized header (DoS guard)', async () => {
    const dir = makeTempDir();
    const filePath = path.join(dir, 'big.safetensors');

    // Claim a header larger than 100 MB
    const lengthBuf = Buffer.alloc(8);
    lengthBuf.writeBigUInt64LE(BigInt(200 * 1024 * 1024), 0);

    fs.writeFileSync(filePath, lengthBuf);

    await expect(parseHeader(filePath)).rejects.toThrow('exceeds maximum');
  });

  it('handles an empty tensor list', async () => {
    const dir = makeTempDir();
    const filePath = createTestSafetensorsFile([], { format: 'pt' }, dir);
    const result = await parseHeader(filePath);

    expect(Object.keys(result.tensors)).toHaveLength(0);
    expect(result.metadata).toEqual({ format: 'pt' });
  });

  it('reads only header bytes, not tensor data', async () => {
    const dir = makeTempDir();
    const bigData = Buffer.alloc(1024 * 1024); // 1 MB of zeros
    const filePath = createTestSafetensorsFile(
      [{ name: 'big_tensor', dtype: 'F32', shape: [256, 1024], data: bigData }],
      {},
      dir,
    );

    const result = await parseHeader(filePath);
    expect(result.tensors['big_tensor']).toBeDefined();
    expect(result.headerLength).toBeLessThan(1024);
  });
});
