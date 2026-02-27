import React, { useState, useMemo, useCallback, useEffect } from 'react';
import ReactFlow, { 
  MiniMap, 
  Controls, 
  Background, 
  useNodesState, 
  useEdgesState, 
  NodeTypes,
  NodeMouseHandler,
  useReactFlow,
  ReactFlowProvider
} from 'reactflow';
import 'reactflow/dist/style.css';

import { ArchitectureNode } from '../../types/tree';
import { LoraAdapterMap } from '../../types/lora';
import { buildGraph } from '../utils/graph';
import { CustomNode } from './CustomNode';
import { Toolbar } from './Toolbar';
import { DetailPanel } from './DetailPanel';

const nodeTypes: NodeTypes = {
  customNode: CustomNode,
};

interface ModelGraphProps {
  tree: ArchitectureNode;
  loraMap: LoraAdapterMap;
  onLoadStats?: (node: ArchitectureNode) => void;
}

function ModelGraphInner({ tree, loraMap, onLoadStats }: ModelGraphProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');
  const [filterMode, setFilterMode] = useState<'highlight' | 'isolate'>('highlight');
  const [filterDtype, setFilterDtype] = useState('all');
  const [filterLora, setFilterLora] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  
  const { fitView } = useReactFlow();

  const handleExpandAll = useCallback(() => {
    const all = new Set<string>();
    function traverse(n: ArchitectureNode) {
      if (n.children.length > 0) {
        all.add(n.fullPath);
        n.children.forEach(traverse);
      }
    }
    traverse(tree);
    setExpandedNodes(all);
  }, [tree]);

  const handleCollapseAll = useCallback(() => {
    setExpandedNodes(new Set());
  }, []);

  const handleNodeDoubleClick: NodeMouseHandler = useCallback((event, node) => {
    setExpandedNodes(prev => {
      const next = new Set(prev);
      if (next.has(node.id)) {
        next.delete(node.id);
      } else {
        next.add(node.id);
      }
      return next;
    });
  }, []);

  const handleNodeClick: NodeMouseHandler = useCallback((event, node) => {
    setSelectedNodeId(node.id);
  }, []);

  useEffect(() => {
    const data = buildGraph(
      tree, 
      loraMap, 
      expandedNodes, 
      searchQuery, 
      filterDtype, 
      filterLora, 
      filterMode
    );
    setNodes(data.nodes);
    setEdges(data.edges);
  }, [tree, loraMap, expandedNodes, searchQuery, filterDtype, filterLora, filterMode, setNodes, setEdges]);

  // Find selected node details
  const selectedNodeData = useMemo(() => {
    if (!selectedNodeId) return null;
    const n = nodes.find(n => n.id === selectedNodeId);
    return n ? n.data.node as ArchitectureNode : null;
  }, [selectedNodeId, nodes]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', width: '100vw' }}>
      <Toolbar 
        onExpandAll={handleExpandAll}
        onCollapseAll={handleCollapseAll}
        searchQuery={searchQuery}
        setSearchQuery={setSearchQuery}
        filterMode={filterMode}
        setFilterMode={setFilterMode}
        filterDtype={filterDtype}
        setFilterDtype={setFilterDtype}
        filterLora={filterLora}
        setFilterLora={setFilterLora}
        onFitView={() => fitView({ duration: 800 })}
      />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            nodeTypes={nodeTypes}
            onNodeDoubleClick={handleNodeDoubleClick}
            onNodeClick={handleNodeClick}
            fitView
            attributionPosition="bottom-right"
          >
            <Background color="#ccc" gap={16} />
            <Controls />
            <MiniMap 
              nodeStrokeColor={(n: any) => {
                if (n.data?.hasLora) return '#89d185';
                if (n.data?.isHighlighted) return '#f0a30a';
                return '#666';
              }}
              nodeColor={(n: any) => {
                return n.data?.isExpanded ? 'rgba(128,128,128,0.1)' : 'var(--vscode-editor-background)';
              }}
              maskColor="rgba(0, 0, 0, 0.2)"
            />
          </ReactFlow>
        </div>
        {selectedNodeData && (
          <DetailPanel 
            node={selectedNodeData} 
            loraMap={loraMap} 
            onClose={() => setSelectedNodeId(null)}
            onLoadStats={onLoadStats}
          />
        )}
      </div>
    </div>
  );
}

export function ModelGraph(props: ModelGraphProps) {
  return (
    <ReactFlowProvider>
      <ModelGraphInner {...props} />
    </ReactFlowProvider>
  );
}
