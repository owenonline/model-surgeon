import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { serialize, SerializationInput } from './serializer';
import { parseHeader } from './headerParser';
import { readTensorData } from './tensorReader';
import {
  createTestSafetensorsFile,
  createFloat32Buffer,
  cleanupTestDir,
} from '../test/helpers/createTestSafetensors';

describe('Safetensors Serializer (R500)', () => {
  let testDir: string;

  beforeEach(() => {
    testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'model-surgeon-serializer-test-'));
  });

  afterEach(() => {
    cleanupTestDir(testDir);
  });

  it('should round-trip parse -> serialize -> parse with identical results', async () => {
    // 1. Create initial test file
    const originalFile = createTestSafetensorsFile(
      [
        {
          name: 'layers.0.attn.weight',
          dtype: 'F32',
          shape: [2, 2],
          data: createFloat32Buffer(4),
        },
        {
          name: 'layers.1.attn.weight',
          dtype: 'F32',
          shape: [4],
          data: createFloat32Buffer(4),
        },
      ],
      { format: 'pt', model: 'test' },
      testDir,
    );

    // 2. Parse original file
    const origHeader = await parseHeader(originalFile);
    
    // 3. Prepare data for serialization
    const input: SerializationInput = {
      metadata: origHeader.metadata,
      tensors: {},
    };

    for (const [name, info] of Object.entries(origHeader.tensors)) {
      const data = await readTensorData(originalFile, origHeader.headerLength, info.dataOffsets);
      input.tensors[name] = {
        dtype: info.dtype,
        shape: info.shape,
        data,
        byteLength: data.length,
      };
    }

    // 4. Serialize to a new file
    const outFile = path.join(testDir, 'output.safetensors');
    await serialize(input, outFile);

    // 5. Parse new file
    const newHeader = await parseHeader(outFile);

    // 6. Compare metadata and tensor properties
    expect(newHeader.metadata).toEqual(origHeader.metadata);
    expect(Object.keys(newHeader.tensors).sort()).toEqual(Object.keys(origHeader.tensors).sort());

    for (const [name, origInfo] of Object.entries(origHeader.tensors)) {
      const newInfo = newHeader.tensors[name];
      expect(newInfo.dtype).toEqual(origInfo.dtype);
      expect(newInfo.shape).toEqual(origInfo.shape);

      // 7. Compare tensor data
      const origData = await readTensorData(originalFile, origHeader.headerLength, origInfo.dataOffsets);
      const newData = await readTensorData(outFile, newHeader.headerLength, newInfo.dataOffsets);
      expect(newData.equals(origData)).toBe(true);
    }
  });
});
