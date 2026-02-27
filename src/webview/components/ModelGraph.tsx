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
import { AlignedComponent } from '../../types/messages';
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
  comparison?: {
    treeB: ArchitectureNode;
    loraMapB: LoraAdapterMap;
    alignedComponents: AlignedComponent[];
  };
  onLoadStats?: (node: ArchitectureNode) => void;
}

function ModelGraphInner({ tree, loraMap, comparison, onLoadStats }: ModelGraphProps) {
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

  const handleNodeDoubleClick: NodeMouseHandler = useCallback((_event, node) => {
    const fullPath = node.data?.node?.fullPath as string | undefined;
    if (!fullPath) return;
    setExpandedNodes(prev => {
      const next = new Set(prev);
      if (next.has(fullPath)) {
        next.delete(fullPath);
      } else {
        next.add(fullPath);
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
      filterMode,
      comparison
    );
    setNodes(data.nodes);
    setEdges(data.edges);
    requestAnimationFrame(() => fitView({ duration: 300 }));
  }, [tree, loraMap, expandedNodes, searchQuery, filterDtype, filterLora, filterMode, comparison, setNodes, setEdges, fitView]);

  // Find selected node details
  const selectedNodeData = useMemo(() => {
    if (!selectedNodeId) return null;
    const n = nodes.find(n => n.id === selectedNodeId);
    return n ? n.data : null;
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
            fitViewOptions={{ padding: 0.2 }}
            panOnDrag
            zoomOnScroll
            zoomOnPinch
            zoomOnDoubleClick={false}
            panOnScroll={false}
            nodesDraggable={false}
            nodesConnectable={false}
            elementsSelectable
            minZoom={0.02}
            maxZoom={4}
            defaultViewport={{ x: 0, y: 0, zoom: 0.5 }}
          >
            <Background color="#444" gap={20} size={1} />
            <Controls showInteractive={false} />
            <MiniMap
              nodeStrokeWidth={0}
              nodeColor={(n: any) => {
                if (n.data?.hasLora) return '#89d185';
                return '#666';
              }}
              maskColor="rgba(0, 0, 0, 0.5)"
            />
          </ReactFlow>
        </div>
        {selectedNodeData && (
          <DetailPanel 
            data={selectedNodeData} 
            loraMap={loraMap} 
            comparison={comparison}
            onClose={() => setSelectedNodeId(null)}
            onLoadStats={(n) => onLoadStats && onLoadStats(n)}
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
