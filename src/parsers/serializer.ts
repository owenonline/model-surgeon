import * as fs from 'fs';
import { SafetensorsDtype, TensorInfo } from '../types/safetensors';

export interface SerializedTensorInfo {
  dtype: SafetensorsDtype;
  shape: number[];
  data?: Buffer;
  dataProvider?: () => Promise<Buffer>;
  byteLength: number;
}

export interface SerializationInput {
  metadata?: Record<string, string>;
  tensors: Record<string, SerializedTensorInfo>;
  onProgress?: (percent: number) => void;
}

/**
 * R500: Safetensors Serializer
 * Writes a complete safetensors file from an in-memory tensor map.
 */
export async function serialize(input: SerializationInput, outputPath: string): Promise<void> {
  const { metadata = {}, tensors } = input;

  // Sort tensor names alphabetically
  const tensorNames = Object.keys(tensors).sort();

  // Compute offsets and prepare header JSON
  const headerObj: any = {
    __metadata__: metadata,
  };

  let currentOffset = 0;
  for (const name of tensorNames) {
    const tensor = tensors[name];
    const dataSize = tensor.byteLength;
    headerObj[name] = {
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_offsets: [currentOffset, currentOffset + dataSize],
    };
    currentOffset += dataSize;
  }

  // Convert header to JSON string and then to Buffer
  const headerJson = JSON.stringify(headerObj);
  const headerBuf = Buffer.from(headerJson, 'utf-8');

  // Ensure header size is 8-byte aligned (not strictly required by standard but good practice or required by some implementations)
  // Actually, standard says:
  // "8 bytes: N, an unsigned little-endian 64-bit integer, containing the size of the header"
  // Let's just follow the python implementation which doesn't necessarily pad but often pads with spaces to 8 bytes.
  // We'll skip padding unless tests require it. Actually huggingface safetensors pad to 8 bytes.
  // Wait, no, python safetensors `serialize` pads the json string with spaces so the tensor data is 8-byte aligned.
  // Let's pad header string.
  // JSON string size is `headerBuf.length`. 8 bytes for N + headerBuf.length.
  // We want `8 + headerBuf.length` to be a multiple of 8.
  // So `headerBuf.length` must be a multiple of 8.
  const paddingLen = (8 - (headerBuf.length % 8)) % 8;
  const paddedHeaderStr = headerJson + ' '.repeat(paddingLen);
  const finalHeaderBuf = Buffer.from(paddedHeaderStr, 'utf-8');

  // Prepare the length buffer
  const lengthBuf = Buffer.alloc(8);
  lengthBuf.writeBigUInt64LE(BigInt(finalHeaderBuf.length), 0);

  // Write to file
  const fh = await fs.promises.open(outputPath, 'w');
  try {
    await fh.write(lengthBuf);
    await fh.write(finalHeaderBuf);
    let totalBytesWritten = 0;
    const totalBytesExpected = currentOffset; // total tensor data bytes

    for (const name of tensorNames) {
      const tensor = tensors[name];
      let buf: Buffer;
      if (tensor.data) {
        buf = tensor.data;
      } else if (tensor.dataProvider) {
        buf = await tensor.dataProvider();
      } else {
        throw new Error(`No data or dataProvider for tensor ${name}`);
      }
      await fh.write(buf);
      
      totalBytesWritten += buf.length;
      if (input.onProgress && totalBytesExpected > 0) {
        input.onProgress((totalBytesWritten / totalBytesExpected) * 100);
      }
    }
  } finally {
    await fh.close();
  }
}
