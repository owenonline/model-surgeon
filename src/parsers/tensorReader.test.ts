import { describe, it, expect, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { parseHeader } from './headerParser';
import { readTensorByName, readTensorData } from './tensorReader';
import {
  createTestSafetensorsFile,
  createFloat32Buffer,
  cleanupTestDir,
} from '../test/helpers/createTestSafetensors';

let testDirs: string[] = [];

function makeTempDir(): string {
  const d = fs.mkdtempSync(path.join(os.tmpdir(), 'reader-test-'));
  testDirs.push(d);
  return d;
}

afterEach(() => {
  for (const d of testDirs) {
    cleanupTestDir(d);
  }
  testDirs = [];
});

describe('R102 -- Lazy Tensor Data Reader', () => {
  it('reads the correct bytes for a single tensor', async () => {
    const dir = makeTempDir();
    const data = createFloat32Buffer(16);
    const filePath = createTestSafetensorsFile(
      [{ name: 'my_tensor', dtype: 'F32', shape: [4, 4], data }],
      {},
      dir,
    );

    const header = await parseHeader(filePath);
    const result = await readTensorByName(filePath, header, 'my_tensor');

    expect(result.length).toBe(data.length);
    expect(result.equals(data)).toBe(true);
  });

  it('reads the correct tensor when multiple exist', async () => {
    const dir = makeTempDir();
    const dataA = createFloat32Buffer(4);
    const dataB = createFloat32Buffer(8);
    const filePath = createTestSafetensorsFile(
      [
        { name: 'tensor_a', dtype: 'F32', shape: [2, 2], data: dataA },
        { name: 'tensor_b', dtype: 'F32', shape: [2, 4], data: dataB },
      ],
      {},
      dir,
    );

    const header = await parseHeader(filePath);

    const resultA = await readTensorByName(filePath, header, 'tensor_a');
    expect(resultA.length).toBe(dataA.length);
    expect(resultA.equals(dataA)).toBe(true);

    const resultB = await readTensorByName(filePath, header, 'tensor_b');
    expect(resultB.length).toBe(dataB.length);
    expect(resultB.equals(dataB)).toBe(true);
  });

  it('throws for a nonexistent tensor name', async () => {
    const dir = makeTempDir();
    const filePath = createTestSafetensorsFile(
      [{ name: 'existing', dtype: 'F32', shape: [2], data: createFloat32Buffer(2) }],
      {},
      dir,
    );

    const header = await parseHeader(filePath);
    await expect(readTensorByName(filePath, header, 'nonexistent')).rejects.toThrow(
      'not found',
    );
  });

  it('uses explicit offset/length (fs.read), not fs.readFile', async () => {
    const dir = makeTempDir();
    const data = createFloat32Buffer(100);
    const filePath = createTestSafetensorsFile(
      [{ name: 't', dtype: 'F32', shape: [100], data }],
      {},
      dir,
    );

    const header = await parseHeader(filePath);
    const offsets = header.tensors['t'].dataOffsets;
    const result = await readTensorData(filePath, header.headerLength, offsets);

    expect(result.length).toBe(data.length);
    expect(result.equals(data)).toBe(true);
  });
});
