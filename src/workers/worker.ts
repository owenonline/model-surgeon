import { parentPort } from 'worker_threads';

interface WorkerTask {
  taskType: string;
  payload: unknown;
}

type TaskHandler = (payload: unknown) => unknown | Promise<unknown>;

const taskHandlers = new Map<string, TaskHandler>();

/**
 * Register a handler for a specific task type.
 * Handlers are added here as new features require worker computation.
 */
taskHandlers.set('ping', () => ({ pong: true }));

taskHandlers.set('echo', (payload) => payload);

parentPort?.on('message', async (task: WorkerTask) => {
  const handler = taskHandlers.get(task.taskType);
  if (!handler) {
    parentPort?.postMessage({
      __error: `Unknown task type: ${task.taskType}`,
      __stack: new Error(`Unknown task type: ${task.taskType}`).stack,
    });
    return;
  }

  try {
    const result = await handler(task.payload);
    parentPort?.postMessage(result);
  } catch (err: unknown) {
    const error = err instanceof Error ? err : new Error(String(err));
    parentPort?.postMessage({
      __error: error.message,
      __stack: error.stack,
    });
  }
});
