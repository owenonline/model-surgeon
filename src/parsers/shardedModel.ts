import * as fs from 'fs';
import * as path from 'path';
import { parseHeader } from './headerParser';
import { UnifiedTensorMap, ShardedTensorInfo } from '../types/safetensors';

interface ShardIndex {
  metadata?: Record<string, string>;
  weight_map: Record<string, string>;
}

/**
 * R103: Detect and handle sharded safetensors models.
 *
 * Looks for model.safetensors.index.json and builds a unified tensor map
 * that routes each tensor to its shard file.
 */
export async function loadShardedModel(indexOrShardPath: string): Promise<UnifiedTensorMap> {
  const dir = path.dirname(indexOrShardPath);
  const indexPath = await findIndexFile(indexOrShardPath, dir);

  if (!indexPath) {
    return loadSingleFileAsUnified(indexOrShardPath);
  }

  const indexContent = await fs.promises.readFile(indexPath, 'utf-8');
  let index: ShardIndex;
  try {
    index = JSON.parse(indexContent);
  } catch {
    throw new Error(`Invalid shard index JSON: ${indexPath}`);
  }

  if (!index.weight_map || typeof index.weight_map !== 'object') {
    throw new Error(`Shard index missing "weight_map": ${indexPath}`);
  }

  const shardFiles = [...new Set(Object.values(index.weight_map))];
  const missingShards: string[] = [];
  for (const shard of shardFiles) {
    const shardPath = path.join(dir, shard);
    try {
      await fs.promises.access(shardPath, fs.constants.R_OK);
    } catch {
      missingShards.push(shard);
    }
  }

  if (missingShards.length > 0) {
    throw new Error(`Missing shard files: ${missingShards.join(', ')}`);
  }

  // Parse all shard headers
  const shardHeaders: Record<string, Awaited<ReturnType<typeof parseHeader>>> = {};
  await Promise.all(
    shardFiles.map(async (shard) => {
      const shardPath = path.join(dir, shard);
      shardHeaders[shard] = await parseHeader(shardPath);
    }),
  );

  // Build the unified tensor map
  const tensors: Record<string, ShardedTensorInfo> = {};
  const shardHeaderLengths: Record<string, number> = {};
  const metadata: Record<string, string> = { ...(index.metadata ?? {}) };

  for (const shard of shardFiles) {
    const header = shardHeaders[shard];
    shardHeaderLengths[shard] = header.headerLength;

    Object.assign(metadata, header.metadata);
  }

  for (const [tensorName, shardFile] of Object.entries(index.weight_map)) {
    const header = shardHeaders[shardFile];
    if (!header) {
      throw new Error(`Shard "${shardFile}" referenced in index but not parsed`);
    }

    const tensorInfo = header.tensors[tensorName];
    if (!tensorInfo) {
      throw new Error(
        `Tensor "${tensorName}" referenced in index but not found in shard "${shardFile}"`,
      );
    }

    tensors[tensorName] = {
      ...tensorInfo,
      shardFile,
    };
  }

  return { metadata, tensors, shardHeaderLengths };
}

/**
 * Wrap a single safetensors file as a unified tensor map for consistent API.
 */
async function loadSingleFileAsUnified(filePath: string): Promise<UnifiedTensorMap> {
  const header = await parseHeader(filePath);
  const fileName = path.basename(filePath);
  const tensors: Record<string, ShardedTensorInfo> = {};

  for (const [name, info] of Object.entries(header.tensors)) {
    tensors[name] = { ...info, shardFile: fileName };
  }

  return {
    metadata: header.metadata,
    tensors,
    shardHeaderLengths: { [fileName]: header.headerLength },
  };
}

async function findIndexFile(
  inputPath: string,
  dir: string,
): Promise<string | null> {
  const baseName = path.basename(inputPath);

  // If the user opened the index file directly
  if (baseName === 'model.safetensors.index.json') {
    return inputPath;
  }

  // Check if an index file exists alongside
  const indexPath = path.join(dir, 'model.safetensors.index.json');
  try {
    await fs.promises.access(indexPath, fs.constants.R_OK);
    return indexPath;
  } catch {
    return null;
  }
}

/**
 * Determine whether a path is a sharded model (has an index file).
 */
export async function isShardedModel(filePath: string): Promise<boolean> {
  const dir = path.dirname(filePath);
  const indexFile = await findIndexFile(filePath, dir);
  return indexFile !== null;
}
