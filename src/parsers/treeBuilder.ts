import { TensorInfo } from '../types/safetensors';
import { LoraAdapterMap } from '../types/lora';
import { ArchitectureNode, NodeType } from '../types/tree';

/**
 * R105: Build a hierarchical architecture tree from a flat tensor map.
 *
 * Parses dot-separated tensor names into a nested structure of
 * blocks, components, and parameters.
 */
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

  // Collect tensor names that are LoRA tensors (to exclude from normal tree)
  const loraTensorNames = new Set<string>();
  for (const pair of Object.values(loraMap)) {
    loraTensorNames.add(pair.loraAName);
    loraTensorNames.add(pair.loraBName);
  }

  // Insert each non-LoRA tensor into the tree
  for (const [name, info] of Object.entries(tensors)) {
    if (loraTensorNames.has(name)) {
      continue;
    }
    insertTensor(root, name, info);
  }

  // Attach LoRA adapters to their parent components
  for (const [baseName, pair] of Object.entries(loraMap)) {
    attachLoraAdapter(root, baseName, pair);
  }

  // Sort children deterministically
  sortTree(root);

  return root;
}

function insertTensor(root: ArchitectureNode, fullName: string, info: TensorInfo): void {
  const segments = fullName.split('.');
  let current = root;

  for (let i = 0; i < segments.length; i++) {
    const segment = segments[i];
    const isLeaf = i === segments.length - 1;
    const currentPath = segments.slice(0, i + 1).join('.');

    if (isLeaf) {
      const leaf: ArchitectureNode = {
        name: segment,
        fullPath: currentPath,
        type: 'parameter',
        children: [],
        tensorInfo: {
          dtype: info.dtype,
          shape: info.shape,
        },
      };
      current.children.push(leaf);
    } else {
      let child = current.children.find((c) => c.name === segment);
      if (!child) {
        const nodeType = classifyNodeType(segment, i, segments.length);
        child = {
          name: segment,
          fullPath: currentPath,
          type: nodeType,
          children: [],
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

function classifyNodeType(segment: string, depth: number, _totalDepth: number): NodeType {
  if (isNumericSegment(segment)) {
    return 'block';
  }
  // Top-level segments and known structural names are components
  if (depth <= 1) {
    return 'component';
  }
  return 'component';
}

function isNumericSegment(segment: string): boolean {
  return /^\d+$/.test(segment);
}

function attachLoraAdapter(
  root: ArchitectureNode,
  baseTensorName: string,
  pair: import('../types/lora').LoraAdapterPair,
): void {
  // Find the parent component of the base tensor
  const segments = baseTensorName.split('.');
  // Walk to the parent (one level above the leaf parameter)
  let current = root;

  for (let i = 0; i < segments.length - 1; i++) {
    const segment = segments[i];
    const child = current.children.find((c) => c.name === segment);
    if (!child) {
      return; // Base tensor path not found in tree (adapter from separate dir)
    }
    current = child;
  }

  // Attach to the parent component
  if (!current.adapters) {
    current.adapters = {};
  }
  current.adapters[baseTensorName] = pair;
}

function sortTree(node: ArchitectureNode): void {
  node.children.sort((a, b) => {
    // Blocks with numeric indices sort by index
    if (a.blockIndex !== undefined && b.blockIndex !== undefined) {
      return a.blockIndex - b.blockIndex;
    }
    // Blocks before non-blocks
    if (a.blockIndex !== undefined) return -1;
    if (b.blockIndex !== undefined) return 1;
    // Alphabetical
    return a.name.localeCompare(b.name);
  });

  for (const child of node.children) {
    sortTree(child);
  }
}
