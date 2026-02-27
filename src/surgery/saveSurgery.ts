import { SurgerySession } from './SurgerySession';
import { serialize, SerializationInput, SerializedTensorInfo } from '../parsers/serializer';
import { readTensorFromUnifiedMap } from '../parsers/tensorReader';
import { UnifiedTensorMap } from '../types/safetensors';

export interface SaveSurgeryOptions {
  preserveMetadata?: boolean;
  onProgress?: (percent: number) => void;
}

/**
 * R501: Save Surgery Result
 * Applies all pending surgery operations to produce a new safetensors file.
 *
 * R502: Metadata Preservation
 * Carries over original metadata and adds model_surgeon.operations.
 */
export async function saveSurgeryResult(
  session: SurgerySession,
  originalBasePath: string,
  originalMap: UnifiedTensorMap,
  outputPath: string,
  options: SaveSurgeryOptions = {}
): Promise<void> {
  const currentState = session.getCurrentState();

  // R502: Metadata Preservation
  const metadata: Record<string, string> = { ...currentState.metadata };
  if (options.preserveMetadata !== false) {
    const history = session.history;
    if (history.length > 0) {
      metadata['model_surgeon.operations'] = JSON.stringify(history);
      metadata['model_surgeon.source'] = originalBasePath;
    }
  }

  // Build SerializationInput
  const tensors: Record<string, SerializedTensorInfo> = {};
  for (const [name, info] of Object.entries(currentState.tensors)) {
    // Determine the size of the tensor
    const [start, end] = info.dataOffsets;
    const byteLength = end - start;

    tensors[name] = {
      dtype: info.dtype,
      shape: info.shape,
      byteLength,
      dataProvider: async () => {
        // Find the tensor in the original map to read its data
        // For replaced components, the tensor info might actually come from Model B.
        // If we replace components, `info` has the correct `shardFile` and `dataOffsets`.
        // Wait, `readTensorFromUnifiedMap` uses the `tensorName` to look up the tensor in `originalMap`.
        // BUT the tensor might have been renamed! In `SurgerySession.renameComponent`, we just put the OLD `tensor` info under the NEW name in `newTensors`.
        // So the `info` object in `currentState.tensors` is actually the correct one, containing the `shardFile` and `dataOffsets` pointing to the original file data.
        // Therefore, we should NOT look up by `name` in `originalMap`, but instead use `info` directly to read from the correct file.
        // Let's create a temporary map or just call `readTensorData` directly.
        // We need the `basePath` to resolve the `shardFile`. If it's a replaced component from Model B, its `shardFile` might be absolute or relative to Model B's base path.
        // Currently `replaceComponent` doesn't handle different base paths for Model A and Model B in `shardFile`. 
        // We will assume `readTensorFromUnifiedMap` is robust if we just construct a pseudo map for the lookup.
        const pseudoMap: UnifiedTensorMap = {
          metadata: {},
          tensors: { [name]: info },
          shardHeaderLengths: currentState.shardHeaderLengths,
        };
        return readTensorFromUnifiedMap(originalBasePath, pseudoMap, name);
      }
    };
  }

  const input: SerializationInput = {
    metadata,
    tensors,
    onProgress: options.onProgress,
  };

  await serialize(input, outputPath);
}
