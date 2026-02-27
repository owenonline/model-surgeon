import { ArchitectureNode } from '../types/tree';
import { AlignedComponent } from '../types/messages';

function getCanonicalPath(path: string): string {
  // Strip PEFT prefixes
  return path.replace(/^base_model\.model\./, '');
}

/**
 * Given two parsed model trees, compute an alignment that matches nodes
 * from Model A to nodes in Model B by their full path.
 */
export function alignArchitectures(treeA: ArchitectureNode, treeB: ArchitectureNode): AlignedComponent[] {
  const mapA = new Map<string, ArchitectureNode>();
  const mapB = new Map<string, ArchitectureNode>();

  function traverse(node: ArchitectureNode, map: Map<string, ArchitectureNode>) {
    if (node.type !== 'root') {
      const canonical = getCanonicalPath(node.fullPath);
      map.set(canonical, node);
    }
    for (const child of node.children) {
      traverse(child, map);
    }
  }

  traverse(treeA, mapA);
  traverse(treeB, mapB);

  const allPaths = new Set([...mapA.keys(), ...mapB.keys()]);
  const alignment: AlignedComponent[] = [];

  for (const path of allPaths) {
    const nodeA = mapA.get(path);
    const nodeB = mapB.get(path);

    if (nodeA && nodeB) {
      let shapeMismatch = false;
      if (nodeA.type === 'parameter' && nodeB.type === 'parameter') {
        const shapeA = nodeA.tensorInfo?.shape;
        const shapeB = nodeB.tensorInfo?.shape;
        if (shapeA && shapeB) {
          if (shapeA.length !== shapeB.length || !shapeA.every((v, i) => v === shapeB[i])) {
            shapeMismatch = true;
          }
        }
      }
      
      alignment.push({
        path,
        status: 'matched',
        ...(shapeMismatch ? { shapeMismatch: true } : {})
      });
    } else if (nodeA) {
      alignment.push({
        path,
        status: 'onlyA'
      });
    } else if (nodeB) {
      alignment.push({
        path,
        status: 'onlyB'
      });
    }
  }

  return alignment;
}
