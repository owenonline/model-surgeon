export interface LoraAdapterPair {
  baseTensorName: string;
  adapterName: string;
  loraAName: string;
  loraBName: string;
  rank: number | null;
  alpha: number | null;
  aShape: number[];
  bShape: number[];
}

export interface AdapterConfig {
  r: number;
  lora_alpha: number;
  target_modules: string[];
  lora_dropout: number;
  [key: string]: unknown;
}

/**
 * Maps a base tensor path (e.g. "backbone.model...k_proj") to all its LoRA adapter pairs.
 * A component can have multiple named adapters (e.g. read_adapter, write_adapter).
 */
export type LoraAdapterMap = Record<string, LoraAdapterPair[]>;
