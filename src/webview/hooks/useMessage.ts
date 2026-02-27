import { useEffect, useCallback, useRef } from 'react';
import {
  ExtensionToWebviewMessage,
  WebviewToExtensionMessage,
  PROTOCOL_VERSION,
} from '../../types/messages';

type MessageHandler<T extends ExtensionToWebviewMessage['type']> = (
  msg: Extract<ExtensionToWebviewMessage, { type: T }>,
) => void;

type HandlerMap = {
  [K in ExtensionToWebviewMessage['type']]?: MessageHandler<K>;
};

declare function acquireVsCodeApi(): {
  postMessage(msg: unknown): void;
  getState(): unknown;
  setState(state: unknown): void;
};

let vscodeApi: ReturnType<typeof acquireVsCodeApi> | null = null;

function getVsCodeApi() {
  if (!vscodeApi) {
    vscodeApi = acquireVsCodeApi();
  }
  return vscodeApi;
}

/**
 * R107 (webview side): React hook that subscribes to incoming messages
 * from the extension host and dispatches them to typed handlers.
 */
export function useMessage(handlers: HandlerMap): void {
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    const listener = (event: MessageEvent) => {
      const msg = event.data as ExtensionToWebviewMessage;
      if (!msg || typeof msg !== 'object' || !('type' in msg)) {
        return;
      }

      if (msg.protocolVersion !== PROTOCOL_VERSION) {
        console.warn(
          `Protocol version mismatch: expected ${PROTOCOL_VERSION}, got ${msg.protocolVersion}`,
        );
        return;
      }

      const handler = handlersRef.current[msg.type] as
        | ((msg: ExtensionToWebviewMessage) => void)
        | undefined;
      if (handler) {
        handler(msg);
      }
    };

    window.addEventListener('message', listener);
    return () => window.removeEventListener('message', listener);
  }, []);
}

/**
 * Send a message from the webview to the extension host.
 */
export function postMessageToExtension(msg: WebviewToExtensionMessage): void {
  const api = getVsCodeApi();
  api.postMessage({ ...msg, protocolVersion: PROTOCOL_VERSION });
}

/**
 * Hook that provides a stable postMessage function.
 */
export function usePostMessage() {
  return useCallback((msg: WebviewToExtensionMessage) => {
    postMessageToExtension(msg);
  }, []);
}
