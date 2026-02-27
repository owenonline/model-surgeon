import * as fs from 'fs';
import * as path from 'path';
import { TensorInfo } from '../types/safetensors';
import { LoraAdapterMap, LoraAdapterPair, AdapterConfig } from '../types/lora';

const LORA_A_SUFFIX = '.lora_A.weight';
const LORA_B_SUFFIX = '.lora_B.weight';
const PEFT_PREFIX = 'base_model.model.';

/**
 * R104: Detect LoRA adapter tensors and associate A/B pairs with base tensors.
 */
export function detectLoraAdapters(
  tensors: Record<string, TensorInfo>,
  adapterConfig?: AdapterConfig | null,
): LoraAdapterMap {
  const loraATensors = new Map<string, { name: string; info: TensorInfo }>();
  const loraBTensors = new Map<string, { name: string; info: TensorInfo }>();

  for (const [name, info] of Object.entries(tensors)) {
    if (name.endsWith(LORA_A_SUFFIX)) {
      const baseName = resolveBaseTensorName(name, LORA_A_SUFFIX);
      loraATensors.set(baseName, { name, info });
    } else if (name.endsWith(LORA_B_SUFFIX)) {
      const baseName = resolveBaseTensorName(name, LORA_B_SUFFIX);
      loraBTensors.set(baseName, { name, info });
    }
  }

  const adapterMap: LoraAdapterMap = {};

  for (const [baseName, aInfo] of loraATensors) {
    const bInfo = loraBTensors.get(baseName);
    if (!bInfo) {
      continue; // Orphan lora_A without matching lora_B
    }

    const pair: LoraAdapterPair = {
      baseTensorName: baseName,
      loraAName: aInfo.name,
      loraBName: bInfo.name,
      rank: adapterConfig?.r ?? inferRank(aInfo.info.shape, bInfo.info.shape),
      alpha: adapterConfig?.lora_alpha ?? null,
      aShape: aInfo.info.shape,
      bShape: bInfo.info.shape,
    };

    adapterMap[baseName] = pair;
  }

  return adapterMap;
}

/**
 * Strip LoRA suffix and PEFT prefix to recover the base tensor name.
 */
function resolveBaseTensorName(loraName: string, suffix: string): string {
  let baseName = loraName.slice(0, -suffix.length);

  // Strip PEFT prefix if present
  if (baseName.startsWith(PEFT_PREFIX)) {
    baseName = baseName.slice(PEFT_PREFIX.length);
  }

  // Re-add .weight suffix since we stripped .lora_A.weight / .lora_B.weight
  return baseName + '.weight';
}

/**
 * Infer LoRA rank from A and B shapes.
 * A shape is typically [rank, in_features], B shape is [out_features, rank].
 */
function inferRank(aShape: number[], bShape: number[]): number | null {
  if (aShape.length >= 2 && bShape.length >= 2) {
    // A: [rank, in], B: [out, rank] — rank is A[0] and B[1]
    if (aShape[0] === bShape[bShape.length - 1]) {
      return aShape[0];
    }
  }
  return null;
}

/**
 * Parse adapter_config.json if it exists near the safetensors files.
 */
export async function parseAdapterConfig(
  safetensorsPath: string,
): Promise<AdapterConfig | null> {
  const dir = path.dirname(safetensorsPath);
  const configPath = path.join(dir, 'adapter_config.json');

  try {
    const content = await fs.promises.readFile(configPath, 'utf-8');
    const config = JSON.parse(content);
    return {
      r: config.r ?? 0,
      lora_alpha: config.lora_alpha ?? 0,
      target_modules: config.target_modules ?? [],
      lora_dropout: config.lora_dropout ?? 0,
      ...config,
    };
  } catch {
    return null;
  }
}

/**
 * Parse adapter_config.json from a specified directory path.
 */
export async function parseAdapterConfigFromDir(
  dirPath: string,
): Promise<AdapterConfig | null> {
  const configPath = path.join(dirPath, 'adapter_config.json');

  try {
    const content = await fs.promises.readFile(configPath, 'utf-8');
    const config = JSON.parse(content);
    return {
      r: config.r ?? 0,
      lora_alpha: config.lora_alpha ?? 0,
      target_modules: config.target_modules ?? [],
      lora_dropout: config.lora_dropout ?? 0,
      ...config,
    };
  } catch {
    return null;
  }
}
