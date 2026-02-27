export interface LoraAdapterPair {
  baseTensorName: string;
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

export type LoraAdapterMap = Record<string, LoraAdapterPair>;
