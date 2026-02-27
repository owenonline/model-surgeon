# Model Surgeon

Visualize, compare, and surgically modify neural network models in [safetensors](https://github.com/huggingface/safetensors) format — without leaving your editor.

## Features

- **Interactive graph view** — explore model architecture as a collapsible node graph with dtype/shape details per tensor
- **Side-by-side comparison** — load two models and see weight differences color-coded by cosine similarity
- **LoRA awareness** — detects and visualizes LoRA adapter pairs; shows rank, alpha, and per-component coverage
- **Surgery tools** — rename components, remove LoRA adapters, or replace components across models, all with undo/redo
- **Non-destructive save** — all edits are staged in a session; write to a new file when ready
- **Lazy I/O** — only the safetensors header is read on open; tensor bytes are fetched on demand

## Installation

### From source (local build)

Prerequisites: [Node.js](https://nodejs.org/) ≥ 18, [Cursor](https://cursor.sh/) or VS Code ≥ 1.85.

```bash
git clone <repo-url>
cd model-surgeon
npm install
npm run build
npm run package          # produces model-surgeon-1.0.0.vsix
```

Install the packaged extension into Cursor:

```bash
cursor --install-extension model-surgeon-1.0.0.vsix
```

Or into VS Code:

```bash
code --install-extension model-surgeon-1.0.0.vsix
```

Then reload your editor window (`Ctrl+Shift+P` → **Developer: Reload Window**).

## Usage

### Open a model

- Open any `.safetensors` file in the Explorer — Model Surgeon opens automatically as the custom editor.
- Or run **Model Surgeon: Open Model** from the Command Palette (`Ctrl+Shift+P`).

### Compare two models

1. Right-click a `.safetensors` file in the Explorer → **Select for Model Surgeon Compare**.
2. Right-click a second file → **Model Surgeon: Compare with Selected**.
3. Or run **Model Surgeon: Open Comparison** from the Command Palette.

### Perform surgery

Right-click any component node in the graph for context menu options:

| Action | Description |
|--------|-------------|
| Rename | Batch-renames all descendant tensors |
| Remove LoRA Adapter | Strips `lora_A` / `lora_B` tensors from a component |
| Rename LoRA Adapter | Updates the adapter key prefix |
| Replace with Model B | Swaps drop-in-compatible components across models (comparison mode only) |

Use `Ctrl+Z` / `Ctrl+Shift+Z` to undo/redo within the webview.

### Save the result

Run **Model Surgeon: Save Surgery Result** or click the save button in the toolbar. A file picker will prompt for the output path (defaults to `<original>_modified.safetensors`).

## Keyboard shortcuts (webview)

| Key | Action |
|-----|--------|
| `Ctrl+F` / `Cmd+F` | Search / filter tensors |
| `Ctrl+Z` | Undo surgery operation |
| `Ctrl+Shift+Z` | Redo surgery operation |
| `Escape` | Clear search filter |

## Development

```bash
npm run watch        # rebuild on file changes (extension + webview)
npm run lint         # ESLint
npm run lint:fix     # ESLint with auto-fix
npm run format       # Prettier
npm run typecheck    # TypeScript type check
npm run test         # Vitest unit tests
```
