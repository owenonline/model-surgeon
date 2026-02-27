import type * as vscode from 'vscode';
import {
  ExtensionToWebviewMessage,
  WebviewToExtensionMessage,
  PROTOCOL_VERSION,
  AnyMessage,
} from '../types/messages';

/**
 * R107: Extension host side of the message protocol.
 *
 * Wraps the vscode.Webview.postMessage API with type validation
 * and protocol versioning.
 */
export class MessageHost {
  private panel: vscode.WebviewPanel | null = null;
  private handlers = new Map<string, (msg: WebviewToExtensionMessage) => void>();
  private disposables: vscode.Disposable[] = [];

  attach(panel: vscode.WebviewPanel): void {
    this.panel = panel;

    const disposable = panel.webview.onDidReceiveMessage(
      (raw: unknown) => {
        const msg = raw as AnyMessage;
        if (!msg || typeof msg !== 'object' || !('type' in msg)) {
          return;
        }

        if ('protocolVersion' in msg && msg.protocolVersion !== PROTOCOL_VERSION) {
          this.postMessage({
            type: 'error',
            protocolVersion: PROTOCOL_VERSION,
            message: `Protocol version mismatch: expected ${PROTOCOL_VERSION}, got ${msg.protocolVersion}. Please reload the webview.`,
            code: 'PROTOCOL_MISMATCH',
          });
          return;
        }

        const handler = this.handlers.get(msg.type);
        if (handler) {
          handler(msg as WebviewToExtensionMessage);
        }
      },
    );

    this.disposables.push(disposable);
  }

  /**
   * Post a validated message to the webview.
   */
  postMessage(message: ExtensionToWebviewMessage): void {
    if (!this.panel) {
      throw new Error('MessageHost: no webview panel attached');
    }

    // Ensure protocol version is set
    const msg = { ...message, protocolVersion: PROTOCOL_VERSION };
    this.panel.webview.postMessage(msg);
  }

  /**
   * Register a handler for a specific incoming message type.
   */
  onMessage<T extends WebviewToExtensionMessage['type']>(
    type: T,
    handler: (msg: Extract<WebviewToExtensionMessage, { type: T }>) => void,
  ): void {
    this.handlers.set(type, handler as (msg: WebviewToExtensionMessage) => void);
  }

  dispose(): void {
    for (const d of this.disposables) {
      d.dispose();
    }
    this.disposables = [];
    this.handlers.clear();
    this.panel = null;
  }
}
