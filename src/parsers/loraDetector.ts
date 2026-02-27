import * as fs from 'fs';
import * as path from 'path';
import { TensorInfo } from '../types/safetensors';
import { LoraAdapterMap, LoraAdapterPair, AdapterConfig } from '../types/lora';

/**
 * Regex patterns for LoRA tensor names. Handles two conventions:
 *
 * 1. Standard PEFT:  *.lora_A.weight / *.lora_B.weight
 * 2. Named adapters: *.lora_A.<adapter_name>.weight / *.lora_B.<adapter_name>.weight
 *
 * Captures: [full match, module_path, "A"|"B", adapter_name|undefined]
 */
const LORA_PATTERN = /^(.+)\.lora_(A|B)(?:\.([^.]+))?\.weight$/;
const BASE_LAYER_PATTERN = /^(.+)\.base_layer\.weight$/;
const PEFT_PREFIX = 'base_model.model.';

export function detectLoraAdapters(
  tensors: Record<string, TensorInfo>,
  adapterConfig?: AdapterConfig | null,
): LoraAdapterMap {
  // Collect all LoRA tensors grouped by (modulePath, adapterName)
  const loraEntries = new Map<
    string,
    { a?: { name: string; info: TensorInfo }; b?: { name: string; info: TensorInfo } }
  >();

  const baseLayerPaths = new Set<string>();

  for (const [name, info] of Object.entries(tensors)) {
    const match = LORA_PATTERN.exec(name);
    if (match) {
      const modulePath = match[1];
      const ab = match[2]; // "A" or "B"
      const adapterName = match[3] ?? 'default';
      const key = `${modulePath}||${adapterName}`;

      let entry = loraEntries.get(key);
      if (!entry) {
        entry = {};
        loraEntries.set(key, entry);
      }
      if (ab === 'A') entry.a = { name, info };
      else entry.b = { name, info };
      continue;
    }

    const baseMatch = BASE_LAYER_PATTERN.exec(name);
    if (baseMatch) {
      baseLayerPaths.add(baseMatch[1]);
    }
  }

  const adapterMap: LoraAdapterMap = {};

  for (const [key, entry] of loraEntries) {
    if (!entry.a || !entry.b) continue;

    const [modulePath, adapterName] = key.split('||');

    let baseTensorName: string;
    if (baseLayerPaths.has(modulePath)) {
      baseTensorName = `${modulePath}.base_layer.weight`;
    } else {
      let stripped = modulePath;
      if (stripped.startsWith(PEFT_PREFIX)) {
        stripped = stripped.slice(PEFT_PREFIX.length);
      }
      baseTensorName = `${stripped}.weight`;
    }

    // Use modulePath as the key for grouping (stable across adapter names)
    const groupKey = modulePath;

    const pair: LoraAdapterPair = {
      baseTensorName,
      adapterName,
      loraAName: entry.a.name,
      loraBName: entry.b.name,
      rank: adapterConfig?.r ?? inferRank(entry.a.info.shape, entry.b.info.shape),
      alpha: adapterConfig?.lora_alpha ?? null,
      aShape: entry.a.info.shape,
      bShape: entry.b.info.shape,
    };

    if (!adapterMap[groupKey]) {
      adapterMap[groupKey] = [];
    }
    adapterMap[groupKey].push(pair);
  }

  return adapterMap;
}

function inferRank(aShape: number[], bShape: number[]): number | null {
  if (aShape.length >= 2 && bShape.length >= 2) {
    if (aShape[0] === bShape[bShape.length - 1]) {
      return aShape[0];
    }
  }
  return null;
}

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
