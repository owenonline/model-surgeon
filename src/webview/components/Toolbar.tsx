import React from 'react';

interface ToolbarProps {
  onExpandAll: () => void;
  onCollapseAll: () => void;
  searchQuery: string;
  setSearchQuery: (q: string) => void;
  filterMode: 'highlight' | 'isolate';
  setFilterMode: (m: 'highlight' | 'isolate') => void;
  filterDtype: string;
  setFilterDtype: (d: string) => void;
  filterLora: boolean;
  setFilterLora: (l: boolean) => void;
  onFitView: () => void;
}

export function Toolbar({
  onExpandAll,
  onCollapseAll,
  searchQuery,
  setSearchQuery,
  filterMode,
  setFilterMode,
  filterDtype,
  setFilterDtype,
  filterLora,
  setFilterLora,
  onFitView
}: ToolbarProps) {
  return (
    <div style={{
      display: 'flex',
      flexWrap: 'wrap',
      gap: '8px',
      padding: '8px',
      backgroundColor: 'var(--vscode-editor-background)',
      borderBottom: '1px solid var(--vscode-panel-border)',
      alignItems: 'center',
      fontSize: '12px'
    }}>
      <div style={{ display: 'flex', gap: '4px' }}>
        <button onClick={onExpandAll}>Expand All</button>
        <button onClick={onCollapseAll}>Collapse All</button>
        <button onClick={onFitView}>Fit View</button>
      </div>
      
      <div style={{ width: '1px', height: '20px', backgroundColor: 'var(--vscode-panel-border)', margin: '0 8px' }} />

      <input 
        type="text" 
        placeholder="Search tensors..." 
        value={searchQuery}
        onChange={e => setSearchQuery(e.target.value)}
        style={{
          background: 'var(--vscode-input-background)',
          color: 'var(--vscode-input-foreground)',
          border: '1px solid var(--vscode-input-border)',
          padding: '4px',
          width: '150px'
        }}
      />

      <select 
        value={filterMode} 
        onChange={e => setFilterMode(e.target.value as any)}
        style={{
          background: 'var(--vscode-dropdown-background)',
          color: 'var(--vscode-dropdown-foreground)',
          border: '1px solid var(--vscode-dropdown-border)',
          padding: '4px'
        }}
      >
        <option value="highlight">Highlight</option>
        <option value="isolate">Isolate</option>
      </select>

      <select 
        value={filterDtype} 
        onChange={e => setFilterDtype(e.target.value)}
        style={{
          background: 'var(--vscode-dropdown-background)',
          color: 'var(--vscode-dropdown-foreground)',
          border: '1px solid var(--vscode-dropdown-border)',
          padding: '4px'
        }}
      >
        <option value="all">All Dtypes</option>
        <option value="F32">FP32</option>
        <option value="F16">FP16</option>
        <option value="BF16">BF16</option>
        <option value="I8">INT8</option>
      </select>

      <label style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
        <input 
          type="checkbox" 
          checked={filterLora}
          onChange={e => setFilterLora(e.target.checked)}
        />
        Has LoRA
      </label>

      <div style={{ flex: 1 }} />
      
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginRight: '8px' }}>
        <div style={{ width: '12px', height: '12px', backgroundColor: '#89d185', borderRadius: '50%' }} />
        <span>LoRA Legend</span>
      </div>
    </div>
  );
}
