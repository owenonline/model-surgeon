import { UnifiedTensorMap, ShardedTensorInfo } from '../types/safetensors';
import { SurgeryOperation } from '../types/messages';

interface SurgeryState {
  tensors: Record<string, ShardedTensorInfo>;
  metadata: Record<string, string>;
  shardHeaderLengths: Record<string, number>;
}

export class SurgerySession {
  private originalState: SurgeryState;
  private states: SurgeryState[] = [];
  private currentIndex: number = -1;

  constructor(initialModel: UnifiedTensorMap) {
    this.originalState = {
      tensors: { ...initialModel.tensors },
      metadata: { ...initialModel.metadata },
      shardHeaderLengths: { ...initialModel.shardHeaderLengths }
    };
    this.states = [this.originalState];
    this.currentIndex = 0;
  }

  public getCurrentState(): UnifiedTensorMap {
    const state = this.states[this.currentIndex];
    return {
      tensors: { ...state.tensors },
      metadata: { ...state.metadata },
      shardHeaderLengths: { ...state.shardHeaderLengths }
    };
  }

  public undo(): boolean {
    if (this.currentIndex > 0) {
      this.currentIndex--;
      return true;
    }
    return false;
  }

  public redo(): boolean {
    if (this.currentIndex < this.states.length - 1) {
      this.currentIndex++;
      return true;
    }
    return false;
  }

  public get pendingChangesCount(): number {
    return this.currentIndex;
  }

  public get history(): SurgeryOperation[] {
     // TODO: To properly support history, we should store operations along with states
     return [];
  }

  private pushState(newState: SurgeryState) {
    this.states = this.states.slice(0, this.currentIndex + 1);
    this.states.push(newState);
    this.currentIndex++;
  }

  /**
   * R401: Rename Component
   * Renames a component (or tensor) by replacing its last path segment with newName.
   * e.g. targetPath = "layers.0.self_attn", newName = "attention"
   * changes all "layers.0.self_attn.*" to "layers.0.attention.*"
   */
  public renameComponent(targetPath: string, newName: string): void {
    const currentState = this.states[this.currentIndex];
    const newTensors: Record<string, ShardedTensorInfo> = {};
    
    // Construct the new base path by replacing the last segment of targetPath
    const parts = targetPath.split('.');
    parts[parts.length - 1] = newName;
    const newBasePath = parts.join('.');
    
    const prefix = targetPath + '.';
    let isSingleTensor = false;
    if (currentState.tensors[targetPath]) {
      isSingleTensor = true;
    }

    for (const [key, tensor] of Object.entries(currentState.tensors)) {
      if (isSingleTensor && key === targetPath) {
        newTensors[newBasePath] = tensor;
      } else if (key.startsWith(prefix) || key === targetPath) {
        const suffix = key === targetPath ? '' : key.substring(prefix.length);
        const newKey = suffix ? `${newBasePath}.${suffix}` : newBasePath;
        newTensors[newKey] = tensor;
      } else {
        newTensors[key] = tensor;
      }
    }

    this.pushState({
      tensors: newTensors,
      metadata: { ...currentState.metadata },
      shardHeaderLengths: { ...currentState.shardHeaderLengths }
    });
  }

  /**
   * R402: Remove LoRA Adapter
   */
  public removeLoraAdapter(targetPath: string): void {
    const currentState = this.states[this.currentIndex];
    const newTensors: Record<string, ShardedTensorInfo> = {};
    
    for (const [key, tensor] of Object.entries(currentState.tensors)) {
      if (key.startsWith(targetPath) && (key.includes('lora_A') || key.includes('lora_B'))) {
        continue;
      }
      newTensors[key] = tensor;
    }

    this.pushState({
      tensors: newTensors,
      metadata: { ...currentState.metadata },
      shardHeaderLengths: { ...currentState.shardHeaderLengths }
    });
  }

  /**
   * R403: Rename LoRA Adapter
   * newName replaces the current adapter prefix inside the LoRA tensor names.
   * e.g. "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
   * typically we replace "base_model.model" if it's the adapter key prefix.
   * Wait, the requirements say: "Rename the LoRA adapter key prefix on a component or set of components."
   */
  public renameLoraAdapter(targetPath: string, newAdapterPrefix: string): void {
    const currentState = this.states[this.currentIndex];
    const newTensors: Record<string, ShardedTensorInfo> = {};
    
    for (const [key, tensor] of Object.entries(currentState.tensors)) {
      if (key.startsWith(targetPath) && (key.includes('lora_A') || key.includes('lora_B'))) {
        // Find the lora_A/lora_B part and replace the segment before it?
        // Actually, if we just want to rename the adapter prefix, a generic replacement 
        // might be replacing the initial prefix if it's peft style, but we're operating on `targetPath`
        // which might just be the component path.
        // A simple implementation: replace the component path prefix with the new prefix? No, it's rename *adapter key prefix*.
        // Often, adapter key prefix is "base_model.model", so if target is "base_model.model.layers.0", new is "my_adapter".
        // Let's do a simple string replace for the specific prefix in the key.
        const newKey = key.replace(targetPath, newAdapterPrefix);
        newTensors[newKey] = tensor;
      } else {
        newTensors[key] = tensor;
      }
    }

    this.pushState({
      tensors: newTensors,
      metadata: { ...currentState.metadata },
      shardHeaderLengths: { ...currentState.shardHeaderLengths }
    });
  }

  /**
   * R404: Replace Component (Cross-Model Surgery)
   */
  public replaceComponent(targetPath: string, sourceModelTensors: Record<string, ShardedTensorInfo>, sourceModelHeaders: Record<string, number>): void {
    const currentState = this.states[this.currentIndex];
    const newTensors: Record<string, ShardedTensorInfo> = { ...currentState.tensors };
    
    const prefix = targetPath + '.';
    for (const key of Object.keys(newTensors)) {
      if (key.startsWith(prefix) || key === targetPath) {
        delete newTensors[key];
      }
    }

    for (const [key, tensor] of Object.entries(sourceModelTensors)) {
      if (key.startsWith(prefix) || key === targetPath) {
        newTensors[key] = tensor;
      }
    }

    this.pushState({
      tensors: newTensors,
      metadata: { ...currentState.metadata },
      shardHeaderLengths: { ...currentState.shardHeaderLengths, ...sourceModelHeaders }
    });
  }

  public removeTensor(targetPath: string): void {
    const currentState = this.states[this.currentIndex];
    const newTensors: Record<string, ShardedTensorInfo> = {};
    
    const prefix = targetPath + '.';
    for (const [key, tensor] of Object.entries(currentState.tensors)) {
      if (key === targetPath || key.startsWith(prefix)) {
        continue;
      }
      newTensors[key] = tensor;
    }

    this.pushState({
      tensors: newTensors,
      metadata: { ...currentState.metadata },
      shardHeaderLengths: { ...currentState.shardHeaderLengths }
    });
  }
}
