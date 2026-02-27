import { SafetensorsDtype } from './safetensors';
import { LoraAdapterPair } from './lora';

export type NodeType = 'root' | 'block' | 'component' | 'parameter';

export interface ArchitectureNode {
  name: string;
  fullPath: string;
  type: NodeType;
  children: ArchitectureNode[];
  tensorInfo?: {
    dtype: SafetensorsDtype;
    shape: number[];
  };
  /** LoRA adapters attached to this component (keyed by adapter name e.g. "default", "read_adapter"). */
  adapters?: Record<string, LoraAdapterPair>;
  /** For block nodes: the numeric index if this is a numbered layer. */
  blockIndex?: number;
  /** Order in which this node first appeared in the tensor file; used to preserve forward-pass order. */
  insertionIndex?: number;
}
