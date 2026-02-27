import { Worker } from 'worker_threads';
import * as path from 'path';

export interface WorkerTask {
  taskType: string;
  payload: unknown;
}

interface PendingTask {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
}

/**
 * R106: A pool of Node.js worker threads for offloading CPU-intensive work.
 *
 * Workers are lazily spawned and reused. The pool manages a configurable
 * number of workers and queues tasks when all are busy.
 */
export class WorkerPool {
  private workers: Worker[] = [];
  private availableWorkers: Worker[] = [];
  private taskQueue: Array<{ task: WorkerTask; pending: PendingTask }> = [];
  private workerPending = new Map<Worker, PendingTask>();
  private maxWorkers: number;
  private workerScript: string;
  private terminated = false;

  constructor(maxWorkers = 2, workerScript?: string) {
    this.maxWorkers = maxWorkers;
    this.workerScript = workerScript ?? path.join(__dirname, 'worker.js');
  }

  /**
   * Execute a task in a worker thread.
   * Returns a promise that resolves with the worker's result or rejects on error.
   */
  async execute<T = unknown>(taskType: string, payload: unknown): Promise<T> {
    if (this.terminated) {
      throw new Error('WorkerPool has been terminated');
    }

    const task: WorkerTask = { taskType, payload };

    return new Promise<T>((resolve, reject) => {
      const pending: PendingTask = {
        resolve: resolve as (value: unknown) => void,
        reject,
      };

      const worker = this.getAvailableWorker();
      if (worker) {
        this.runTask(worker, task, pending);
      } else {
        this.taskQueue.push({ task, pending });
      }
    });
  }

  /**
   * Terminate all workers. Called when the extension deactivates.
   */
  async terminate(): Promise<void> {
    this.terminated = true;

    // Reject all queued tasks
    for (const { pending } of this.taskQueue) {
      pending.reject(new Error('WorkerPool terminated'));
    }
    this.taskQueue = [];

    // Reject all in-flight tasks before terminating workers
    for (const [, pending] of this.workerPending) {
      pending.reject(new Error('WorkerPool terminated'));
    }
    this.workerPending.clear();

    // Terminate all workers
    const terminations = this.workers.map((w) => w.terminate());
    await Promise.all(terminations);
    this.workers = [];
    this.availableWorkers = [];
  }

  private getAvailableWorker(): Worker | null {
    if (this.availableWorkers.length > 0) {
      return this.availableWorkers.pop()!;
    }

    // Spawn a new worker if under the limit
    if (this.workers.length < this.maxWorkers) {
      return this.spawnWorker();
    }

    return null;
  }

  private spawnWorker(): Worker {
    const worker = new Worker(this.workerScript);
    this.workers.push(worker);

    worker.on('message', (result) => {
      const pending = this.workerPending.get(worker);
      this.workerPending.delete(worker);

      if (pending) {
        if (result && typeof result === 'object' && result.__error) {
          const err = new Error(result.__error);
          if (result.__stack) {
            err.stack = result.__stack;
          }
          pending.reject(err);
        } else {
          pending.resolve(result);
        }
      }

      this.onWorkerFree(worker);
    });

    worker.on('error', (err) => {
      const pending = this.workerPending.get(worker);
      this.workerPending.delete(worker);

      if (pending) {
        pending.reject(err);
      }

      // Remove broken worker and replace
      this.removeWorker(worker);
      this.processQueue();
    });

    worker.on('exit', (code) => {
      if (code !== 0 && !this.terminated) {
        const pending = this.workerPending.get(worker);
        this.workerPending.delete(worker);

        if (pending) {
          pending.reject(new Error(`Worker exited with code ${code}`));
        }
        this.removeWorker(worker);
        this.processQueue();
      }
    });

    return worker;
  }

  private runTask(worker: Worker, task: WorkerTask, pending: PendingTask): void {
    this.workerPending.set(worker, pending);
    worker.postMessage(task);
  }

  private onWorkerFree(worker: Worker): void {
    if (this.taskQueue.length > 0) {
      const { task, pending } = this.taskQueue.shift()!;
      this.runTask(worker, task, pending);
    } else {
      this.availableWorkers.push(worker);
    }
  }

  private removeWorker(worker: Worker): void {
    const idx = this.workers.indexOf(worker);
    if (idx >= 0) {
      this.workers.splice(idx, 1);
    }
    const availIdx = this.availableWorkers.indexOf(worker);
    if (availIdx >= 0) {
      this.availableWorkers.splice(availIdx, 1);
    }
  }

  private processQueue(): void {
    while (this.taskQueue.length > 0) {
      const worker = this.getAvailableWorker();
      if (!worker) break;
      const { task, pending } = this.taskQueue.shift()!;
      this.runTask(worker, task, pending);
    }
  }
}
