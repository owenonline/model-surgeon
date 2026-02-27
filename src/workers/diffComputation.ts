import { readTensorData } from '../parsers/tensorReader';
import { SafetensorsDtype, DTYPE_BYTE_SIZE } from '../types/safetensors';

export interface DiffMetrics {
  cosineSimilarity: number;
  l2NormDiff: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
}

export interface ComputeDiffPayload {
  pathA: string;
  headerLengthA: number;
  offsetsA: [number, number];
  dtypeA: SafetensorsDtype;

  pathB: string;
  headerLengthB: number;
  offsetsB: [number, number];
  dtypeB: SafetensorsDtype;
}

// Float16 to Float32 conversion
function f16ToF32(val: number): number {
  const sign = (val & 0x8000) >> 15;
  const exp = (val & 0x7c00) >> 10;
  const frac = val & 0x03ff;

  if (exp === 0) {
    if (frac === 0) return sign ? -0 : 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024);
  }
  if (exp === 0x1f) {
    if (frac === 0) return sign ? -Infinity : Infinity;
    return NaN;
  }
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024);
}

// BFloat16 to Float32 conversion
function bf16ToF32(val: number): number {
  // BFloat16 is just the top 16 bits of a Float32
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setUint16(2, val, true); // Little endian? safetensors is little-endian
  // Actually, standard float32 is read properly if we set the bits in the correct endianness.
  // We can just construct a float32 directly by shifting.
  const int32 = val << 16;
  view.setInt32(0, int32, true);
  return view.getFloat32(0, true);
}

function parseBuffer(buffer: Buffer, dtype: SafetensorsDtype): Float32Array {
  const elementSize = DTYPE_BYTE_SIZE[dtype];
  const numElements = buffer.length / elementSize;
  const result = new Float32Array(numElements);

  if (dtype === 'F32') {
    const view = new Float32Array(buffer.buffer, buffer.byteOffset, numElements);
    result.set(view);
    return result;
  }

  const dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  for (let i = 0; i < numElements; i++) {
    switch (dtype) {
      case 'F64': result[i] = dataView.getFloat64(i * 8, true); break;
      case 'F16': result[i] = f16ToF32(dataView.getUint16(i * 2, true)); break;
      case 'BF16': result[i] = bf16ToF32(dataView.getUint16(i * 2, true)); break;
      case 'I64': result[i] = Number(dataView.getBigInt64(i * 8, true)); break;
      case 'I32': result[i] = dataView.getInt32(i * 4, true); break;
      case 'I16': result[i] = dataView.getInt16(i * 2, true); break;
      case 'I8':  result[i] = dataView.getInt8(i); break;
      case 'U8':  result[i] = dataView.getUint8(i); break;
      case 'BOOL': result[i] = dataView.getUint8(i) !== 0 ? 1 : 0; break;
      default:
        // F8 variants are complex, default to 0 for simplicity in this implementation
        result[i] = 0;
    }
  }

  return result;
}

export async function computeDiff(payload: ComputeDiffPayload): Promise<DiffMetrics> {
  const [bufA, bufB] = await Promise.all([
    readTensorData(payload.pathA, payload.headerLengthA, payload.offsetsA),
    readTensorData(payload.pathB, payload.headerLengthB, payload.offsetsB)
  ]);

  const arrA = parseBuffer(bufA, payload.dtypeA);
  const arrB = parseBuffer(bufB, payload.dtypeB);

  // If shapes mismatched heavily, arrays might be different sizes.
  // Requirement says for matched components. If sizes differ, diff up to minimum length.
  const len = Math.min(arrA.length, arrB.length);

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  let sumSqDiff = 0;
  let maxAbsDiff = 0;
  let sumAbsDiff = 0;

  for (let i = 0; i < len; i++) {
    const a = arrA[i];
    const b = arrB[i];
    const diff = a - b;
    const absDiff = Math.abs(diff);

    dotProduct += a * b;
    normA += a * a;
    normB += b * b;
    
    sumSqDiff += diff * diff;
    if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
    sumAbsDiff += absDiff;
  }

  const magA = Math.sqrt(normA);
  const magB = Math.sqrt(normB);
  
  let cosineSimilarity = 0;
  if (magA > 0 && magB > 0) {
    cosineSimilarity = dotProduct / (magA * magB);
  } else if (magA === 0 && magB === 0) {
    cosineSimilarity = 1; // Both zero vectors
  }

  return {
    cosineSimilarity,
    l2NormDiff: Math.sqrt(sumSqDiff),
    maxAbsDiff,
    meanAbsDiff: len > 0 ? sumAbsDiff / len : 0
  };
}
