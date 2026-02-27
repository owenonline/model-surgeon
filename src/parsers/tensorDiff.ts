import * as fs from 'fs';
import { ShardedTensorInfo, DTYPE_BYTE_SIZE } from '../types/safetensors';
import { TensorDiffMetrics } from '../types/messages';

/**
 * Maximum number of elements per tensor we will eagerly diff.
 * 1M elements × 2 bytes (BF16) = 2 MB per tensor.
 * This covers all typical LoRA weights (rank × hidden_dim) while skipping
 * large base weight matrices.
 */
const DIFF_ELEMENT_LIMIT = 1_000_000;

// ─── Float conversion helpers ─────────────────────────────────────────────────

// Reusable 4-byte scratch buffer for bit-level float32 reinterpretation.
const _scratch = new ArrayBuffer(4);
const _scratchView = new DataView(_scratch);

function bf16ToFloat32(u16: number): number {
  // BF16 occupies the upper 16 bits of float32.
  // Shift left 16 to reconstruct the float32 bit pattern.
  _scratchView.setUint32(0, u16 << 16, true); // little-endian
  return _scratchView.getFloat32(0, true);
}

function float16ToFloat32(h: number): number {
  const sign = (h & 0x8000) >>> 15;
  const exp = (h & 0x7c00) >>> 10;
  const mant = h & 0x03ff;

  let bits: number;
  if (exp === 0) {
    if (mant === 0) {
      bits = sign << 31;
    } else {
      // Denormalized — normalize it
      let e = 0;
      let m = mant;
      while ((m & 0x400) === 0) {
        m <<= 1;
        e++;
      }
      bits = (sign << 31) | ((112 - e + 1) << 23) | ((m & 0x3ff) << 13);
    }
  } else if (exp === 31) {
    // Inf or NaN
    bits = (sign << 31) | 0x7f800000 | (mant << 13);
  } else {
    bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  }

  _scratchView.setInt32(0, bits, true);
  return _scratchView.getFloat32(0, true);
}

function bufferToFloat32(buf: Buffer, dtype: string): Float32Array {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const bpe = DTYPE_BYTE_SIZE[dtype as keyof typeof DTYPE_BYTE_SIZE] ?? 2;
  const n = buf.byteLength / bpe;
  const out = new Float32Array(n);

  switch (dtype) {
    case 'F32':
      for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
      break;
    case 'F64':
      for (let i = 0; i < n; i++) out[i] = dv.getFloat64(i * 8, true);
      break;
    case 'BF16':
      for (let i = 0; i < n; i++) out[i] = bf16ToFloat32(dv.getUint16(i * 2, true));
      break;
    case 'F16':
      for (let i = 0; i < n; i++) out[i] = float16ToFloat32(dv.getUint16(i * 2, true));
      break;
    case 'I32':
      for (let i = 0; i < n; i++) out[i] = dv.getInt32(i * 4, true);
      break;
    case 'I16':
      for (let i = 0; i < n; i++) out[i] = dv.getInt16(i * 2, true);
      break;
    case 'I8':
      for (let i = 0; i < n; i++) out[i] = dv.getInt8(i);
      break;
    case 'U8':
      for (let i = 0; i < n; i++) out[i] = dv.getUint8(i);
      break;
    default:
      // Unknown dtype — leave as zeros
      break;
  }

  return out;
}

// ─── Batched tensor reader ────────────────────────────────────────────────────

/**
 * Read multiple tensors from potentially multiple shard files efficiently.
 * Opens each unique shard file only once.
 */
async function readTensorsBatched(
  tensors: Record<string, ShardedTensorInfo>,
  headerLengths: Record<string, number>,
  names: string[],
): Promise<Map<string, Float32Array>> {
  // Group tensor names by shard file
  const byFile = new Map<string, string[]>();
  for (const name of names) {
    const info = tensors[name];
    if (!info) continue;
    const group = byFile.get(info.shardFile) ?? [];
    group.push(name);
    byFile.set(info.shardFile, group);
  }

  const result = new Map<string, Float32Array>();

  for (const [shardFile, tensorNames] of byFile) {
    const headerLen = headerLengths[shardFile] ?? 0;
    const fd = await fs.promises.open(shardFile, 'r');
    try {
      for (const name of tensorNames) {
        const info = tensors[name];
        const byteStart = 8 + headerLen + info.dataOffsets[0];
        const byteLen = info.dataOffsets[1] - info.dataOffsets[0];
        const buf = Buffer.alloc(byteLen);
        await fd.read(buf, 0, byteLen, byteStart);
        result.set(name, bufferToFloat32(buf, info.dtype));
      }
    } finally {
      await fd.close();
    }
  }

  return result;
}

// ─── Diff computation ─────────────────────────────────────────────────────────

function computeMetrics(a: Float32Array, b: Float32Array): TensorDiffMetrics {
  const n = Math.min(a.length, b.length);

  let dotAB = 0;
  let normA2 = 0;
  let normB2 = 0;
  let sumAbsDiff = 0;
  let sumSqDiff = 0;
  let maxAbsDiff = 0;

  for (let i = 0; i < n; i++) {
    const ai = a[i];
    const bi = b[i];
    dotAB += ai * bi;
    normA2 += ai * ai;
    normB2 += bi * bi;
    const diff = ai - bi;
    const absDiff = diff < 0 ? -diff : diff;
    if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
    sumAbsDiff += absDiff;
    sumSqDiff += diff * diff;
  }

  const denom = Math.sqrt(normA2) * Math.sqrt(normB2);
  const rawCos = denom < 1e-12 ? (normA2 < 1e-24 && normB2 < 1e-24 ? 1 : 0) : dotAB / denom;
  const cosineSimilarity = Math.max(-1, Math.min(1, rawCos));

  return {
    cosineSimilarity,
    l2NormDiff: Math.sqrt(sumSqDiff),
    maxAbsDiff,
    meanAbsDiff: n > 0 ? sumAbsDiff / n : 0,
  };
}

// ─── Public API ───────────────────────────────────────────────────────────────

function tensorElementCount(info: ShardedTensorInfo): number {
  return info.shape.reduce((acc, d) => acc * d, 1);
}

/**
 * For every path in `matchedPaths`, read both tensors (A and B) and compute
 * diff metrics. Tensors exceeding DIFF_ELEMENT_LIMIT are skipped.
 *
 * Returns a map from canonical path → TensorDiffMetrics.
 */
export async function computeTensorDiffs(
  tensorsA: Record<string, ShardedTensorInfo>,
  headerLengthsA: Record<string, number>,
  tensorsB: Record<string, ShardedTensorInfo>,
  headerLengthsB: Record<string, number>,
  matchedPaths: string[],
): Promise<Map<string, TensorDiffMetrics>> {
  // Filter to paths that exist in both models and are small enough
  const eligible = matchedPaths.filter((path) => {
    const infoA = tensorsA[path];
    const infoB = tensorsB[path];
    if (!infoA || !infoB) return false;
    return tensorElementCount(infoA) <= DIFF_ELEMENT_LIMIT;
  });

  if (eligible.length === 0) return new Map();

  // Read all eligible tensors from both models (batched by shard file)
  const [dataA, dataB] = await Promise.all([
    readTensorsBatched(tensorsA, headerLengthsA, eligible),
    readTensorsBatched(tensorsB, headerLengthsB, eligible),
  ]);

  const result = new Map<string, TensorDiffMetrics>();

  for (const path of eligible) {
    const fa = dataA.get(path);
    const fb = dataB.get(path);
    if (!fa || !fb) continue;
    result.set(path, computeMetrics(fa, fb));
  }

  return result;
}
