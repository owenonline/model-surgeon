import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { SurgerySession } from './SurgerySession';
import { saveSurgeryResult } from './saveSurgery';
import { parseHeader } from '../parsers/headerParser';
import { readTensorByName } from '../parsers/tensorReader';
import { UnifiedTensorMap } from '../types/safetensors';
import {
  createTestSafetensorsFile,
  createFloat32Buffer,
  cleanupTestDir,
} from '../test/helpers/createTestSafetensors';

describe('Save Surgery Result (R501 & R502)', () => {
  let testDir: string;
  let originalFile: string;
  let originalMap: UnifiedTensorMap;

  beforeEach(async () => {
    testDir = fs.mkdtempSync(path.join(os.tmpdir(), 'model-surgeon-save-test-'));
    originalFile = createTestSafetensorsFile(
      [
        {
          name: 'layers.0.self_attn.q_proj.weight',
          dtype: 'F32',
          shape: [2, 2],
          data: createFloat32Buffer(4),
        },
        {
          name: 'layers.0.self_attn.k_proj.weight',
          dtype: 'F32',
          shape: [2, 2],
          data: createFloat32Buffer(4),
        },
      ],
      { format: 'pt' },
      testDir,
    );

    const header = await parseHeader(originalFile);
    originalMap = {
      metadata: header.metadata,
      tensors: {},
      shardHeaderLengths: { 'model.safetensors': header.headerLength },
    };

    for (const [name, info] of Object.entries(header.tensors)) {
      originalMap.tensors[name] = {
        ...info,
        shardFile: 'model.safetensors',
      };
    }
  });

  afterEach(() => {
    cleanupTestDir(testDir);
  });

  it('should save renamed component and preserve metadata', async () => {
    const session = new SurgerySession(originalMap);
    
    // Perform a rename operation
    session.renameComponent('layers.0.self_attn.q_proj', 'query_proj');

    const outFile = path.join(testDir, 'output.safetensors');
    
    let progressReached100 = false;
    await saveSurgeryResult(session, originalFile, originalMap, outFile, {
      onProgress: (p) => {
        if (p === 100) progressReached100 = true;
      }
    });

    expect(progressReached100).toBe(true);

    const newHeader = await parseHeader(outFile);
    
    // Check renamed tensor
    expect(newHeader.tensors['layers.0.self_attn.query_proj.weight']).toBeDefined();
    expect(newHeader.tensors['layers.0.self_attn.q_proj.weight']).toBeUndefined();
    // Check untouched tensor
    expect(newHeader.tensors['layers.0.self_attn.k_proj.weight']).toBeDefined();

    // Check data of renamed tensor
    const origData = await readTensorByName(originalFile, await parseHeader(originalFile), 'layers.0.self_attn.q_proj.weight');
    const newData = await readTensorByName(outFile, newHeader, 'layers.0.self_attn.query_proj.weight');
    expect(newData.equals(origData)).toBe(true);

    // R502: Check metadata
    expect(newHeader.metadata['format']).toBe('pt');
    expect(newHeader.metadata['model_surgeon.operations']).toBeDefined();
    expect(newHeader.metadata['model_surgeon.source']).toBe(originalFile);
    
    const ops = JSON.parse(newHeader.metadata['model_surgeon.operations']);
    expect(ops.length).toBe(1);
    expect(ops[0].operationType).toBe('renameTensor');
    expect(ops[0].targetPath).toBe('layers.0.self_attn.q_proj');
    expect(ops[0].newName).toBe('query_proj');
  });

  it('should omit removed tensors in output', async () => {
    const session = new SurgerySession(originalMap);
    
    // Perform a remove operation
    session.removeTensor('layers.0.self_attn.k_proj.weight');

    const outFile = path.join(testDir, 'output_removed.safetensors');
    await saveSurgeryResult(session, originalFile, originalMap, outFile);

    const newHeader = await parseHeader(outFile);
    
    expect(newHeader.tensors['layers.0.self_attn.k_proj.weight']).toBeUndefined();
    expect(newHeader.tensors['layers.0.self_attn.q_proj.weight']).toBeDefined();
  });
});
