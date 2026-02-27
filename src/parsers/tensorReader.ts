import * as fs from 'fs';
import * as path from 'path';
import { ParsedHeader, ShardedTensorInfo, UnifiedTensorMap } from '../types/safetensors';

/**
 * R102: Lazily read a single tensor's raw bytes from disk.
 *
 * Uses fs.read() with explicit offset/length to read exactly the bytes
 * for the requested tensor, without loading the full file.
 */
export async function readTensorData(
  filePath: string,
  headerLength: number,
  dataOffsets: [number, number],
): Promise<Buffer> {
  const [start, end] = dataOffsets;
  const byteLength = end - start;
  const fileOffset = 8 + headerLength + start;

  const fd = await fs.promises.open(filePath, 'r');
  try {
    const buffer = Buffer.alloc(byteLength);
    const { bytesRead } = await fd.read(buffer, 0, byteLength, fileOffset);
    if (bytesRead < byteLength) {
      throw new Error(
        `Failed to read tensor data: expected ${byteLength} bytes, got ${bytesRead}`,
      );
    }
    return buffer;
  } finally {
    await fd.close();
  }
}

/**
 * Read tensor data from a parsed single-file model by tensor name.
 */
export async function readTensorByName(
  filePath: string,
  header: ParsedHeader,
  tensorName: string,
): Promise<Buffer> {
  const tensorInfo = header.tensors[tensorName];
  if (!tensorInfo) {
    throw new Error(`Tensor "${tensorName}" not found in header`);
  }
  return readTensorData(filePath, header.headerLength, tensorInfo.dataOffsets);
}

/**
 * Read tensor data from a unified (possibly sharded) tensor map by tensor name.
 */
export async function readTensorFromUnifiedMap(
  basePath: string,
  tensorMap: UnifiedTensorMap,
  tensorName: string,
): Promise<Buffer> {
  const tensorInfo: ShardedTensorInfo | undefined = tensorMap.tensors[tensorName];
  if (!tensorInfo) {
    throw new Error(`Tensor "${tensorName}" not found in tensor map`);
  }

  const shardPath = resolvePath(basePath, tensorInfo.shardFile);
  const headerLength = tensorMap.shardHeaderLengths[tensorInfo.shardFile];

  if (headerLength === undefined) {
    throw new Error(`Header length not found for shard "${tensorInfo.shardFile}"`);
  }

  return readTensorData(shardPath, headerLength, tensorInfo.dataOffsets);
}

function resolvePath(basePath: string, filename: string): string {
  const dir = path.dirname(basePath);
  return path.join(dir, filename);
}
