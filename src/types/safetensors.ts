/**
 * All dtype values supported by the safetensors format.
 */
export type SafetensorsDtype =
  | 'F64'
  | 'F32'
  | 'F16'
  | 'BF16'
  | 'I64'
  | 'I32'
  | 'I16'
  | 'I8'
  | 'U8'
  | 'BOOL'
  | 'F8_E4M3'
  | 'F8_E5M2';

export const VALID_DTYPES: ReadonlySet<string> = new Set<string>([
  'F64',
  'F32',
  'F16',
  'BF16',
  'I64',
  'I32',
  'I16',
  'I8',
  'U8',
  'BOOL',
  'F8_E4M3',
  'F8_E5M2',
]);

/** Bytes per element for each dtype. */
export const DTYPE_BYTE_SIZE: Record<SafetensorsDtype, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
  F8_E4M3: 1,
  F8_E5M2: 1,
};

export interface TensorInfo {
  dtype: SafetensorsDtype;
  shape: number[];
  dataOffsets: [number, number];
}

export interface ParsedHeader {
  metadata: Record<string, string>;
  tensors: Record<string, TensorInfo>;
  headerLength: number;
}

/**
 * Extended tensor info that includes which shard file the tensor resides in.
 * Used for sharded models.
 */
export interface ShardedTensorInfo extends TensorInfo {
  shardFile: string;
}

export interface UnifiedTensorMap {
  metadata: Record<string, string>;
  tensors: Record<string, ShardedTensorInfo>;
  /** Maps shard filenames to their parsed header lengths. */
  shardHeaderLengths: Record<string, number>;
}
