import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';

export interface TestTensor {
  name: string;
  dtype: string;
  shape: number[];
  data: Buffer;
}

/**
 * Create a valid safetensors file on disk for testing.
 * Follows the binary format: 8-byte LE header length + JSON header + tensor data.
 */
export function createTestSafetensorsFile(
  tensors: TestTensor[],
  metadata: Record<string, string> = {},
  dir?: string,
): string {
  const targetDir = dir ?? fs.mkdtempSync(path.join(os.tmpdir(), 'model-surgeon-test-'));
  const filePath = path.join(targetDir, 'model.safetensors');

  const { headerBuf, dataBuf } = buildSafetensorsBuffer(tensors, metadata);

  const lengthBuf = Buffer.alloc(8);
  lengthBuf.writeBigUInt64LE(BigInt(headerBuf.length), 0);

  const fullBuf = Buffer.concat([lengthBuf, headerBuf, dataBuf]);
  fs.writeFileSync(filePath, fullBuf);

  return filePath;
}

/**
 * Create a named safetensors shard file.
 */
export function createTestShardFile(
  fileName: string,
  tensors: TestTensor[],
  metadata: Record<string, string> = {},
  dir: string,
): string {
  const filePath = path.join(dir, fileName);

  const { headerBuf, dataBuf } = buildSafetensorsBuffer(tensors, metadata);

  const lengthBuf = Buffer.alloc(8);
  lengthBuf.writeBigUInt64LE(BigInt(headerBuf.length), 0);

  fs.writeFileSync(filePath, Buffer.concat([lengthBuf, headerBuf, dataBuf]));
  return filePath;
}

/**
 * Create a model.safetensors.index.json file.
 */
export function createTestIndexFile(
  weightMap: Record<string, string>,
  dir: string,
  metadata: Record<string, string> = {},
): string {
  const indexPath = path.join(dir, 'model.safetensors.index.json');
  const indexContent = {
    metadata,
    weight_map: weightMap,
  };
  fs.writeFileSync(indexPath, JSON.stringify(indexContent, null, 2));
  return indexPath;
}

function buildSafetensorsBuffer(
  tensors: TestTensor[],
  metadata: Record<string, string>,
): { headerBuf: Buffer; dataBuf: Buffer } {
  // Sort tensors alphabetically
  const sorted = [...tensors].sort((a, b) => a.name.localeCompare(b.name));

  // Build header JSON
  const headerObj: Record<string, unknown> = {};

  if (Object.keys(metadata).length > 0) {
    headerObj['__metadata__'] = metadata;
  }

  let offset = 0;
  for (const t of sorted) {
    headerObj[t.name] = {
      dtype: t.dtype,
      shape: t.shape,
      data_offsets: [offset, offset + t.data.length],
    };
    offset += t.data.length;
  }

  const headerBuf = Buffer.from(JSON.stringify(headerObj), 'utf-8');

  // Concatenate tensor data in order
  const dataBuf = Buffer.concat(sorted.map((t) => t.data));

  return { headerBuf, dataBuf };
}

/**
 * Create a simple float32 tensor buffer with the given number of elements.
 */
export function createFloat32Buffer(numElements: number): Buffer {
  const buf = Buffer.alloc(numElements * 4);
  for (let i = 0; i < numElements; i++) {
    buf.writeFloatLE(i * 0.1, i * 4);
  }
  return buf;
}

/**
 * Create a test adapter_config.json file.
 */
export function createTestAdapterConfig(
  dir: string,
  config: {
    r?: number;
    lora_alpha?: number;
    target_modules?: string[];
    lora_dropout?: number;
  } = {},
): string {
  const configPath = path.join(dir, 'adapter_config.json');
  const content = {
    r: config.r ?? 8,
    lora_alpha: config.lora_alpha ?? 16,
    target_modules: config.target_modules ?? ['q_proj', 'v_proj'],
    lora_dropout: config.lora_dropout ?? 0.05,
  };
  fs.writeFileSync(configPath, JSON.stringify(content, null, 2));
  return configPath;
}

/**
 * Clean up a test directory.
 */
export function cleanupTestDir(dirPath: string): void {
  try {
    fs.rmSync(dirPath, { recursive: true, force: true });
  } catch {
    // Best-effort cleanup
  }
}
