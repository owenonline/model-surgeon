import { describe, it, expect, afterEach } from 'vitest';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import { WorkerPool } from './workerPool';

function createTestWorkerScript(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'wp-test-'));
  const scriptPath = path.join(dir, 'worker.js');
  const code = `
const { parentPort } = require('worker_threads');

parentPort.on('message', async (task) => {
  try {
    if (task.taskType === 'ping') {
      parentPort.postMessage({ pong: true });
    } else if (task.taskType === 'echo') {
      parentPort.postMessage(task.payload);
    } else if (task.taskType === 'error') {
      throw new Error('Intentional worker error');
    } else if (task.taskType === 'slow') {
      await new Promise(r => setTimeout(r, 100));
      parentPort.postMessage({ done: true });
    } else {
      parentPort.postMessage({ __error: 'Unknown task: ' + task.taskType });
    }
  } catch (err) {
    parentPort.postMessage({ __error: err.message, __stack: err.stack });
  }
});
`;
  fs.writeFileSync(scriptPath, code);
  return scriptPath;
}

let pools: WorkerPool[] = [];
let tmpDirs: string[] = [];

afterEach(async () => {
  for (const pool of pools) {
    await pool.terminate();
  }
  pools = [];
  for (const d of tmpDirs) {
    try { fs.rmSync(d, { recursive: true, force: true }); } catch { /* best-effort cleanup */ }
  }
  tmpDirs = [];
});

function createPool(maxWorkers = 2): WorkerPool {
  const script = createTestWorkerScript();
  tmpDirs.push(path.dirname(script));
  const pool = new WorkerPool(maxWorkers, script);
  pools.push(pool);
  return pool;
}

describe('R106 -- Worker Thread Infrastructure', () => {
  it('executes a simple task and returns the result', async () => {
    const pool = createPool();
    const result = await pool.execute<{ pong: boolean }>('ping', {});
    expect(result.pong).toBe(true);
  });

  it('echoes payload back', async () => {
    const pool = createPool();
    const payload = { hello: 'world', num: 42 };
    const result = await pool.execute('echo', payload);
    expect(result).toEqual(payload);
  });

  it('surfaces worker errors as rejected promises', async () => {
    const pool = createPool();
    await expect(pool.execute('error', {})).rejects.toThrow('Intentional worker error');
  });

  it('processes concurrent tasks with limited workers', async () => {
    const pool = createPool(2);

    const results = await Promise.all([
      pool.execute('echo', { id: 1 }),
      pool.execute('echo', { id: 2 }),
      pool.execute('echo', { id: 3 }),
      pool.execute('echo', { id: 4 }),
    ]);

    expect(results).toEqual([
      { id: 1 },
      { id: 2 },
      { id: 3 },
      { id: 4 },
    ]);
  });

  it('rejects queued tasks on terminate', async () => {
    const pool = createPool(1);

    // Start a slow task to occupy the worker; catch its rejection
    const slow = pool.execute('slow', {}).catch(() => 'rejected');
    // Queue a second task
    const queued = pool.execute('echo', { late: true }).catch((e: Error) => e.message);

    // Terminate immediately
    await pool.terminate();

    // Both tasks should be rejected
    expect(await slow).toBe('rejected');
    expect(await queued).toContain('terminated');
  });

  it('rejects new tasks after termination', async () => {
    const pool = createPool();
    await pool.terminate();
    await expect(pool.execute('ping', {})).rejects.toThrow('terminated');
  });
});
