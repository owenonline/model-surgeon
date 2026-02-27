import * as fs from 'fs';
import { ParsedHeader, TensorInfo, VALID_DTYPES, SafetensorsDtype } from '../types/safetensors';

const MAX_HEADER_SIZE = 100 * 1024 * 1024; // 100 MB DoS guard

/**
 * R101: Parse a safetensors file header without loading tensor data.
 *
 * Binary format: 8-byte LE uint64 header length, UTF-8 JSON header, raw tensor bytes.
 * Only reads the first 8 + headerLength bytes.
 */
export async function parseHeader(filePath: string): Promise<ParsedHeader> {
  const fd = await fs.promises.open(filePath, 'r');
  try {
    // Read the 8-byte header length
    const lengthBuf = Buffer.alloc(8);
    const { bytesRead } = await fd.read(lengthBuf, 0, 8, 0);
    if (bytesRead < 8) {
      throw new Error(`Invalid safetensors file: could not read header length (got ${bytesRead} bytes)`);
    }

    const headerLength = Number(lengthBuf.readBigUInt64LE(0));

    if (headerLength <= 0) {
      throw new Error('Invalid safetensors file: header length is zero or negative');
    }

    if (headerLength > MAX_HEADER_SIZE) {
      throw new Error(
        `Header size ${headerLength} bytes exceeds maximum allowed ${MAX_HEADER_SIZE} bytes`,
      );
    }

    // Read the JSON header
    const headerBuf = Buffer.alloc(headerLength);
    const headerRead = await fd.read(headerBuf, 0, headerLength, 8);
    if (headerRead.bytesRead < headerLength) {
      throw new Error(
        `Invalid safetensors file: could not read full header (expected ${headerLength}, got ${headerRead.bytesRead})`,
      );
    }

    const headerJson = headerBuf.toString('utf-8');
    let rawHeader: Record<string, unknown>;
    try {
      rawHeader = JSON.parse(headerJson);
    } catch {
      throw new Error('Invalid safetensors file: header is not valid JSON');
    }

    const metadata: Record<string, string> = {};
    const tensors: Record<string, TensorInfo> = {};

    for (const [key, value] of Object.entries(rawHeader)) {
      if (key === '__metadata__') {
        if (value && typeof value === 'object') {
          for (const [mk, mv] of Object.entries(value as Record<string, unknown>)) {
            metadata[mk] = String(mv);
          }
        }
        continue;
      }

      const tensorData = value as {
        dtype?: string;
        shape?: number[];
        data_offsets?: [number, number];
      };

      if (!tensorData.dtype || !VALID_DTYPES.has(tensorData.dtype)) {
        throw new Error(`Invalid dtype "${tensorData.dtype}" for tensor "${key}"`);
      }

      if (!Array.isArray(tensorData.shape)) {
        throw new Error(`Invalid shape for tensor "${key}"`);
      }

      if (
        !Array.isArray(tensorData.data_offsets) ||
        tensorData.data_offsets.length !== 2 ||
        typeof tensorData.data_offsets[0] !== 'number' ||
        typeof tensorData.data_offsets[1] !== 'number'
      ) {
        throw new Error(`Invalid data_offsets for tensor "${key}"`);
      }

      tensors[key] = {
        dtype: tensorData.dtype as SafetensorsDtype,
        shape: tensorData.shape,
        dataOffsets: tensorData.data_offsets,
      };
    }

    return { metadata, tensors, headerLength };
  } finally {
    await fd.close();
  }
}
