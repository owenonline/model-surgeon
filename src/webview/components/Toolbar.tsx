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
      
      <button 
        onClick={() => {
          const overlay = document.getElementById('help-overlay');
          if (overlay) overlay.style.display = 'flex';
        }}
        style={{
          background: 'none',
          border: '1px solid var(--vscode-button-secondaryBorder)',
          color: 'var(--vscode-button-secondaryForeground)',
          backgroundColor: 'var(--vscode-button-secondaryBackground)',
          borderRadius: '2px',
          padding: '2px 8px',
          cursor: 'pointer'
        }}
        aria-label="Open Help Quick Reference"
      >
        ? Help
      </button>

      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', marginRight: '8px', marginLeft: '8px' }}>
        <div style={{ width: '12px', height: '12px', backgroundColor: 'var(--vscode-testing-iconPassed, #89d185)', borderRadius: '50%' }} />
        <span>LoRA Legend</span>
      </div>
      
      {/* Help Overlay - Hidden by default */}
      <div 
        id="help-overlay"
        style={{
          display: 'none',
          position: 'fixed',
          top: 0, left: 0, right: 0, bottom: 0,
          backgroundColor: 'rgba(0,0,0,0.5)',
          zIndex: 1000,
          justifyContent: 'center',
          alignItems: 'center'
        }}
        onClick={(e) => {
          if (e.target === e.currentTarget) {
            e.currentTarget.style.display = 'none';
          }
        }}
      >
        <div style={{
          backgroundColor: 'var(--vscode-editor-background)',
          color: 'var(--vscode-editor-foreground)',
          border: '1px solid var(--vscode-panel-border)',
          borderRadius: '4px',
          padding: '24px',
          maxWidth: '500px',
          boxShadow: '0 4px 6px rgba(0,0,0,0.3)',
          position: 'relative'
        }}>
          <button 
            onClick={() => {
              const overlay = document.getElementById('help-overlay');
              if (overlay) overlay.style.display = 'none';
            }}
            style={{
              position: 'absolute', top: '8px', right: '8px',
              background: 'none', border: 'none', color: 'var(--vscode-foreground)',
              cursor: 'pointer', fontSize: '16px'
            }}
            aria-label="Close Help"
          >
            ✕
          </button>
          <h2 style={{ marginTop: 0, borderBottom: '1px solid var(--vscode-panel-border)', paddingBottom: '8px' }}>Model Surgeon Quick Reference</h2>
          
          <h3 style={{ fontSize: '14px', marginBottom: '4px' }}>Keyboard Shortcuts</h3>
          <ul style={{ margin: '0 0 16px 0', paddingLeft: '20px', fontSize: '13px' }}>
            <li><kbd>Ctrl/Cmd + F</kbd> - Focus search bar</li>
            <li><kbd>Double Click</kbd> - Collapse/expand node</li>
            <li><kbd>Ctrl/Cmd + Z</kbd> - Undo last surgery</li>
            <li><kbd>Ctrl/Cmd + Shift + Z</kbd> - Redo surgery</li>
          </ul>

          <h3 style={{ fontSize: '14px', marginBottom: '4px' }}>Node Colors & Indicators</h3>
          <ul style={{ margin: '0 0 16px 0', paddingLeft: '20px', fontSize: '13px', listStyle: 'none' }}>
            <li style={{ marginBottom: '4px' }}><span style={{ display: 'inline-block', width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--vscode-testing-iconPassed, #89d185)', marginRight: '8px' }}></span>Node has LoRA adapter</li>
            <li style={{ marginBottom: '4px' }}><span style={{ display: 'inline-block', width: '12px', height: '12px', border: '2px solid var(--vscode-testing-iconPassed, #28a745)', marginRight: '8px' }}></span>Comparison: Identical / High similarity</li>
            <li style={{ marginBottom: '4px' }}><span style={{ display: 'inline-block', width: '12px', height: '12px', border: '2px solid var(--vscode-testing-iconQueued, #ffc107)', marginRight: '8px' }}></span>Comparison: Moderate diff</li>
            <li style={{ marginBottom: '4px' }}><span style={{ display: 'inline-block', width: '12px', height: '12px', border: '2px solid var(--vscode-testing-iconFailed, #dc3545)', marginRight: '8px' }}></span>Comparison: High diff</li>
            <li style={{ marginBottom: '4px' }}><span style={{ display: 'inline-block', width: '12px', height: '12px', border: '2px dashed var(--vscode-descriptionForeground, #888)', marginRight: '8px' }}></span>Comparison: Absent in other model</li>
          </ul>

          <h3 style={{ fontSize: '14px', marginBottom: '4px' }}>Surgery Workflow</h3>
          <ol style={{ margin: 0, paddingLeft: '20px', fontSize: '13px' }}>
            <li>Right-click nodes to rename or remove tensors.</li>
            <li>In comparison mode, right-click to replace with the other model's version.</li>
            <li>Pending operations queue up in the session.</li>
            <li>Run <strong>Save Surgery Result</strong> to apply changes to a new file.</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
