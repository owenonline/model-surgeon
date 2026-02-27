import React, { useState, useEffect, useRef } from 'react';
import { ArchitectureNode } from '../../types/tree';

interface RenameModalProps {
  node: ArchitectureNode;
  onConfirm: (newName: string) => void;
  onCancel: () => void;
}

export function RenameModal({ node, onConfirm, onCancel }: RenameModalProps) {
  const [value, setValue] = useState(node.name);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.select();
  }, []);

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && value.trim()) {
      onConfirm(value.trim());
    } else if (e.key === 'Escape') {
      onCancel();
    }
    e.stopPropagation();
  }

  return (
    <div
      style={{
        position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
        backgroundColor: 'rgba(0,0,0,0.5)',
        zIndex: 10000,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onCancel(); }}
    >
      <div style={{
        backgroundColor: 'var(--vscode-editor-background)',
        border: '1px solid var(--vscode-panel-border)',
        borderRadius: '4px',
        padding: '20px',
        width: '360px',
        boxShadow: '0 4px 16px rgba(0,0,0,0.4)',
        fontFamily: 'var(--vscode-font-family)',
      }}>
        <h3 style={{ margin: '0 0 8px', fontSize: '14px', color: 'var(--vscode-editor-foreground)' }}>
          Rename Component
        </h3>
        <div style={{ fontSize: '12px', color: 'var(--vscode-descriptionForeground)', marginBottom: '12px', wordBreak: 'break-all' }}>
          {node.fullPath}
        </div>
        <input
          ref={inputRef}
          type="text"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          style={{
            width: '100%',
            boxSizing: 'border-box',
            padding: '6px 8px',
            background: 'var(--vscode-input-background)',
            color: 'var(--vscode-input-foreground)',
            border: '1px solid var(--vscode-focusBorder, #007acc)',
            borderRadius: '2px',
            fontSize: '13px',
            outline: 'none',
          }}
          spellCheck={false}
        />
        <div style={{ display: 'flex', gap: '8px', marginTop: '12px', justifyContent: 'flex-end' }}>
          <button
            onClick={onCancel}
            style={{
              padding: '4px 12px', fontSize: '13px', cursor: 'pointer',
              background: 'var(--vscode-button-secondaryBackground)',
              color: 'var(--vscode-button-secondaryForeground)',
              border: '1px solid var(--vscode-button-secondaryBorder, transparent)',
              borderRadius: '2px',
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => value.trim() && onConfirm(value.trim())}
            disabled={!value.trim() || value.trim() === node.name}
            style={{
              padding: '4px 12px', fontSize: '13px', cursor: 'pointer',
              background: 'var(--vscode-button-background)',
              color: 'var(--vscode-button-foreground)',
              border: 'none',
              borderRadius: '2px',
              opacity: (!value.trim() || value.trim() === node.name) ? 0.5 : 1,
            }}
          >
            Rename
          </button>
        </div>
      </div>
    </div>
  );
}
