import { ArchitectureNode } from './tree';
import { LoraAdapterMap } from './lora';

export const PROTOCOL_VERSION = 1;

export interface LoadModelMessage {
  type: 'loadModel';
  protocolVersion: number;
  filePath: string;
}

export interface ModelLoadedMessage {
  type: 'modelLoaded';
  protocolVersion: number;
  tree: ArchitectureNode;
  loraMap: LoraAdapterMap;
  tensorCount: number;
  filePath: string;
}

export interface LoadComparisonMessage {
  type: 'loadComparison';
  protocolVersion: number;
  filePath: string;
}

export interface ComparisonResultMessage {
  type: 'comparisonResult';
  protocolVersion: number;
  alignedComponents: AlignedComponent[];
  treeB: ArchitectureNode;
  loraMapB: LoraAdapterMap;
  filePathB: string;
}

export interface TensorDiffMetrics {
  /** Cosine similarity between the two weight vectors: 1 = identical direction, 0 = orthogonal. */
  cosineSimilarity: number;
  /** L2 norm of (A - B). */
  l2NormDiff: number;
  /** Maximum per-element absolute difference. */
  maxAbsDiff: number;
  /** Mean per-element absolute difference. */
  meanAbsDiff: number;
}

export interface AlignedComponent {
  path: string;
  status: 'matched' | 'onlyA' | 'onlyB';
  shapeMismatch?: boolean;
  /** Populated for matched parameter nodes that are small enough to read eagerly. */
  diffMetrics?: TensorDiffMetrics;
}

export interface PerformSurgeryMessage {
  type: 'performSurgery';
  protocolVersion: number;
  operation: SurgeryOperation;
}

export interface SurgeryOperation {
  operationType: 'renameTensor' | 'removeTensor' | 'replaceTensor' | 'renameLoraAdapter' | 'removeLoraAdapter';
  targetPath: string;
  newName?: string;
  sourceModel?: 'A' | 'B';
}

export interface SurgeryResultMessage {
  type: 'surgeryResult';
  protocolVersion: number;
  success: boolean;
  updatedTree?: ArchitectureNode;
  error?: string;
}

export interface ErrorMessage {
  type: 'error';
  protocolVersion: number;
  message: string;
  code?: string;
}

export interface ProgressMessage {
  type: 'progress';
  protocolVersion: number;
  taskId: string;
  label: string;
  percent: number;
}

export type ExtensionToWebviewMessage =
  | ModelLoadedMessage
  | ComparisonResultMessage
  | SurgeryResultMessage
  | ErrorMessage
  | ProgressMessage;

export type WebviewToExtensionMessage =
  | LoadModelMessage
  | LoadComparisonMessage
  | PerformSurgeryMessage;

export type AnyMessage = ExtensionToWebviewMessage | WebviewToExtensionMessage;
