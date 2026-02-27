import { TensorInfo } from '../types/safetensors';
import { LoraAdapterMap, LoraAdapterPair } from '../types/lora';
import { ArchitectureNode, NodeType } from '../types/tree';

const LORA_PATTERN = /\.lora_(A|B)(?:\.[^.]+)?\.weight$/;
const BASE_LAYER_PATTERN = /\.base_layer\.weight$/;

export function buildArchitectureTree(
  tensors: Record<string, TensorInfo>,
  loraMap: LoraAdapterMap = {},
): ArchitectureNode {
  const root: ArchitectureNode = {
    name: 'model',
    fullPath: '',
    type: 'root',
    children: [],
  };

  const loraTensorNames = new Set<string>();
  for (const pairs of Object.values(loraMap)) {
    for (const pair of pairs) {
      loraTensorNames.add(pair.loraAName);
      loraTensorNames.add(pair.loraBName);
    }
  }

  // Counter shared across all insertTensor calls so first-seen order is preserved.
  let insertionCounter = 0;

  for (const [name, info] of Object.entries(tensors)) {
    if (loraTensorNames.has(name)) continue;
    if (LORA_PATTERN.test(name)) continue;
    insertTensor(root, name, info, insertionCounter++);
  }

  for (const [modulePath, pairs] of Object.entries(loraMap)) {
    attachLoraAdapters(root, modulePath, pairs);
  }

  sortTree(root);
  return root;
}

function insertTensor(
  root: ArchitectureNode,
  fullName: string,
  info: TensorInfo,
  insertionCounter: number,
): void {
  const segments = fullName.split('.');
  let current = root;

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const isLeaf = i === segments.length - 1;
    const currentPath = segments.slice(0, i + 1).join('.');

    if (isLeaf) {
      current.children.push({
        name: segment,
        fullPath: currentPath,
        type: 'parameter',
        children: [],
        tensorInfo: { dtype: info.dtype, shape: info.shape },
        insertionIndex: insertionCounter,
      });
    } else {
      let child = current.children.find((c) => c.name === segment);
      if (!child) {
        const nodeType = classifyNodeType(segment);
        child = {
          name: segment,
          fullPath: currentPath,
          type: nodeType,
          children: [],
          // Record insertionIndex only once, when the node is first created.
          insertionIndex: insertionCounter,
        };
        if (nodeType === 'block' && isNumericSegment(segment)) {
          child.blockIndex = parseInt(segment, 10);
        }
        current.children.push(child);
      }
      current = child;
    }
  }
}

function classifyNodeType(segment: string): NodeType {
  if (isNumericSegment(segment)) return 'block';
  return 'component';
}

function isNumericSegment(segment: string): boolean {
  return /^\d+$/.test(segment);
}

function attachLoraAdapters(
  root: ArchitectureNode,
  modulePath: string,
  pairs: LoraAdapterPair[],
): void {
  const segments = modulePath.split('.');
  let current = root;

  for (const segment of segments) {
    const child = current.children.find((c) => c.name === segment);
    if (!child) return;
    current = child;
  }

  if (!current.adapters) {
    current.adapters = {};
  }
  for (const pair of pairs) {
    current.adapters[pair.adapterName] = pair;
  }
}

function sortTree(node: ArchitectureNode): void {
  node.children.sort((a, b) => {
    // Numeric block nodes always sort by their block index.
    if (a.blockIndex !== undefined && b.blockIndex !== undefined) {
      return a.blockIndex - b.blockIndex;
    }
    if (a.blockIndex !== undefined) return -1;
    if (b.blockIndex !== undefined) return 1;

    // For named components, use the order they first appeared in the file
    // (forward-pass / module-registration order) rather than alphabetical order.
    const ai = a.insertionIndex ?? 0;
    const bi = b.insertionIndex ?? 0;
    return ai - bi;
  });

  for (const child of node.children) {
    sortTree(child);
  }
}
