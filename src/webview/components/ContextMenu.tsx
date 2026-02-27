import React, { useEffect, useRef } from 'react';
import { ArchitectureNode } from '../../types/tree';

export interface ContextMenuState {
  x: number;
  y: number;
  node: ArchitectureNode;
  modelId: 'A' | 'B';
  hasLora: boolean;
  hasComparison: boolean;
}

interface ContextMenuProps extends ContextMenuState {
  onClose: () => void;
  onRename: (node: ArchitectureNode) => void;
  onRemove: (node: ArchitectureNode) => void;
  onRemoveLora: (node: ArchitectureNode) => void;
  onReplaceFromB: (node: ArchitectureNode) => void;
}

const menuItemStyle: React.CSSProperties = {
  padding: '6px 14px',
  cursor: 'pointer',
  fontSize: '13px',
  color: 'var(--vscode-menu-foreground)',
  whiteSpace: 'nowrap',
  userSelect: 'none',
};

const menuItemDangerStyle: React.CSSProperties = {
  ...menuItemStyle,
  color: 'var(--vscode-errorForeground, #f48771)',
};

const separatorStyle: React.CSSProperties = {
  height: '1px',
  backgroundColor: 'var(--vscode-menu-separatorBackground, rgba(255,255,255,0.1))',
  margin: '4px 0',
};

export function ContextMenu({
  x, y, node, modelId, hasLora, hasComparison,
  onClose, onRename, onRemove, onRemoveLora, onReplaceFromB,
}: ContextMenuProps) {
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        onClose();
      }
    }
    function handleKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose();
    }
    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  // Clamp menu position so it doesn't go off screen
  const menuWidth = 220;
  const menuHeight = 160;
  const clampedX = Math.min(x, window.innerWidth - menuWidth - 8);
  const clampedY = Math.min(y, window.innerHeight - menuHeight - 8);

  const isParameter = node.type === 'parameter';

  return (
    <div
      ref={menuRef}
      style={{
        position: 'fixed',
        top: clampedY,
        left: clampedX,
        zIndex: 9999,
        backgroundColor: 'var(--vscode-menu-background)',
        border: '1px solid var(--vscode-menu-border, rgba(255,255,255,0.12))',
        borderRadius: '4px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.4)',
        minWidth: `${menuWidth}px`,
        paddingTop: '4px',
        paddingBottom: '4px',
        fontFamily: 'var(--vscode-font-family)',
      }}
      onContextMenu={e => e.preventDefault()}
    >
      {/* Header */}
      <div style={{
        padding: '4px 14px 6px',
        fontSize: '11px',
        color: 'var(--vscode-descriptionForeground)',
        borderBottom: '1px solid var(--vscode-menu-separatorBackground, rgba(255,255,255,0.1))',
        marginBottom: '4px',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        whiteSpace: 'nowrap',
        maxWidth: `${menuWidth - 4}px`,
      }}
        title={node.fullPath}
      >
        {node.fullPath || node.name}
      </div>

      {/* Rename — available for components and blocks, not parameters */}
      {!isParameter && (
        <MenuItem
          label="Rename…"
          icon="✏️"
          onClick={() => { onClose(); onRename(node); }}
        />
      )}

      {/* Remove */}
      <MenuItem
        label={isParameter ? 'Remove Tensor' : 'Remove Component'}
        icon="🗑️"
        danger
        onClick={() => { onClose(); onRemove(node); }}
      />

      {/* Remove LoRA */}
      {hasLora && (
        <>
          <div style={separatorStyle} />
          <MenuItem
            label="Remove LoRA Adapters"
            icon="⚡"
            danger
            onClick={() => { onClose(); onRemoveLora(node); }}
          />
        </>
      )}

      {/* Replace from Model B */}
      {hasComparison && modelId === 'A' && !isParameter && (
        <>
          <div style={separatorStyle} />
          <MenuItem
            label="Replace with Model B version"
            icon="🔄"
            onClick={() => { onClose(); onReplaceFromB(node); }}
          />
        </>
      )}
    </div>
  );
}

function MenuItem({
  label, icon, danger, onClick,
}: {
  label: string;
  icon: string;
  danger?: boolean;
  onClick: () => void;
}) {
  const [hovered, setHovered] = React.useState(false);

  return (
    <div
      role="menuitem"
      style={{
        ...(danger ? menuItemDangerStyle : menuItemStyle),
        backgroundColor: hovered ? 'var(--vscode-menu-selectionBackground, rgba(255,255,255,0.08))' : 'transparent',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
      }}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={onClick}
    >
      <span style={{ fontSize: '12px', width: '16px', textAlign: 'center', flexShrink: 0 }}>{icon}</span>
      {label}
    </div>
  );
}
