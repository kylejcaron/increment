import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { cp, mkdir, mkdtemp, readFile, readdir, realpath, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Miniflare } from 'miniflare';
import { chromium } from 'playwright';

const here = dirname(fileURLToPath(import.meta.url));
const root = join(here, '../..');
const vectorsPath = join(root, 'docs/reference/assignment-v1-vectors.json');
const sdk = join(root, 'sdks/assignment-js');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const env = { ...process.env };
for (const name of ['NODE_PATH', 'PYTHONPATH', 'PYTHONHOME', 'UV_PROJECT_ENVIRONMENT', 'VIRTUAL_ENV']) {
  delete env[name];
}

const json = async (file) => JSON.parse(await readFile(file, 'utf8'));

function run(program, args, cwd, capture = false) {
  return execFileSync(program, args, { cwd, env, encoding: 'utf8', stdio: capture ? ['ignore', 'pipe', 'inherit'] : 'inherit' });
}

function assertCases(vectors, actual, runtime) {
  assert.equal(actual.length, vectors.cases.length, runtime);
  for (let i = 0; i < actual.length; i++) {
    const fixture = vectors.cases[i];
    const expected = 'error' in fixture
      ? { id: fixture.id, error: fixture.error }
      : { id: fixture.id, result: fixture.expected };
    assert.deepEqual(actual[i], expected, `${runtime}: ${fixture.id}`);
  }
}

function assertExample(actual, vectors) {
  const expect = (id) => vectors.cases.find((row) => row.id === id).expected;
  const assigned = expect('range-start-inclusive');
  const gap = {
    ...assigned, membership: expect('entry-eligible'), status: 'unallocated',
    experiment: null, arm: null, layer_bps: null, enrollment_bps: null, arm_bps: null,
  }; // acct:3 is eligible (4018), in the checkout gap (8649).
  const expected = [
    { unit_id: 'acct:1', decision: assigned },
    { unit_id: 'acct:2', decision: expect('holdout-before-checkout') },
    { unit_id: 'acct:3', decision: gap },
    { unit_id: 'acct:1', decision: expect('eligible-miss-other-layer') },
    { unit_id: 'acct:1', decision: expect('eligible-standalone') },
  ];
  assert.deepEqual(actual.assignments, expected);
  assert.deepEqual(actual.memberships, [
    { unit_id: 'acct:1', membership: expect('entry-eligible') },
    { unit_id: 'acct:2', membership: expect('entry-held_out') },
    { unit_id: 'acct:3', membership: expect('entry-eligible') },
  ]);
  assert.deepEqual(actual.responses, expected.map(({ unit_id, decision }) => ({
    unit_id, target: decision.target, content: decision.arm === 'treatment' ? 'New experience' : 'Baseline experience',
  })));
  assert.deepEqual(actual.ramp, { before: assigned, after: expect('ramp-preserves-cohort-and-arm') });
}

async function installArtifacts(scratch) {
  run('uv', ['build', '--wheel', '--out-dir', scratch, root], root);
  const wheels = (await readdir(scratch)).filter((file) => file.endsWith('.whl'));
  assert.equal(wheels.length, 1, 'Expected exactly one wheel');
  const venv = join(scratch, 'venv');
  const python = join(venv, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  run('uv', ['venv', '--python', env.ASSIGNMENT_PYTHON ?? '3.12', venv], scratch);
  run('uv', ['pip', 'install', '--python', python, join(scratch, wheels[0])], scratch);

  run(npm, ['ci', '--no-audit', '--no-fund'], sdk);
  run(npm, ['run', 'build'], sdk);
  const [packed] = JSON.parse(run(npm, ['pack', '--json', '--ignore-scripts', '--pack-destination', scratch], sdk, true));
  const consumer = join(scratch, 'consumer');
  await mkdir(consumer);
  run(npm, ['install', '--ignore-scripts', '--no-audit', '--no-fund', join(scratch, packed.filename)], consumer);
  const packageDir = join(consumer, 'node_modules/@increment/assignment');
  assert.equal(await realpath(packageDir), packageDir, 'Expected packed files, not a link to source');
  return { python, consumer, packageDir, packed, wheel: wheels[0] };
}

async function qualifyWorker(consumer, packageDir, vectors) {
  const dist = join(packageDir, 'dist');
  const modules = [join(consumer, 'worker.mjs'), join(consumer, 'run_cases.mjs')];
  for (const file of await readdir(dist)) {
    if (file.endsWith('.js')) modules.push(join(dist, file));
  }
  const worker = new Miniflare({
    compatibilityDate: '2026-07-30',
    modulesRoot: consumer,
    modules: await Promise.all(modules.map(async (path) => ({
      type: 'ESModule', path, contents: await readFile(path, 'utf8'),
    }))),
  });
  try {
    const response = await worker.dispatchFetch('http://assignment.local/', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(vectors),
    });
    const body = await response.text();
    assert.equal(response.status, 200, body);
    assertCases(vectors, JSON.parse(body), 'workerd');
    const { version } = await json(join(here, 'node_modules/workerd/package.json'));
    console.log(`workerd ${version}: ${vectors.cases.length} cases`);
  } finally {
    await worker.dispose();
  }
}

async function qualifyBrowser(consumer, vectors) {
  const { serve } = await import(pathToFileURL(join(consumer, 'examples/server.mjs')));
  const { url, close } = await serve();
  try {
    const browser = await chromium.launch();
    try {
      const page = await browser.newPage();
      const sdkRequests = [];
      page.on('request', (request) => {
        if (new URL(request.url()).pathname.startsWith('/sdk/')) sdkRequests.push(request.url());
      });
      for (const [route, id] of [['server', 'holdout-before-checkout'], ['client', 'range-start-inclusive']]) {
        const expected = vectors.cases.find((fixture) => fixture.id === id).expected;
        const response = await page.goto(`${url}/${route}`);
        assert.equal(response.status(), 200);
        assert.equal(await page.evaluate(() => isSecureContext), true);
        const result = page.locator('output[data-status]');
        await result.waitFor({ timeout: 10_000 });
        assert.equal(await result.textContent(), `${expected.status}: ${expected.arm ?? 'baseline'}`);
        if (route === 'server') {
          assert.deepEqual(JSON.parse(await page.locator('#server-decision').textContent()), expected);
          assert.equal(sdkRequests.length, 0, 'Server decision page must not load an evaluator');
        } else {
          assert.ok(sdkRequests.some((request) => new URL(request).pathname === '/sdk/index.js'));
        }
      }
      console.log(`Chromium ${browser.version()}: rendered the server decision and independently evaluated the client-only ID`);
    } finally {
      await browser.close();
    }
  } finally {
    await close();
  }
}

const scratch = await realpath(await mkdtemp(join(tmpdir(), 'increment-assignment-')));
try {
  const vectors = await json(vectorsPath);
  const { python, consumer, packageDir, packed, wheel } = await installArtifacts(scratch);
  await cp(join(root, 'examples/assignment'), join(consumer, 'examples'), { recursive: true });
  for (const file of ['worker.mjs', 'run_cases.mjs', 'consumer.mts']) {
    await cp(join(here, file), join(consumer, file));
  }
  run(python, ['-I', join(here, 'check_python.py'), vectorsPath], consumer);
  const nodeResults = JSON.parse(run(process.execPath, ['--input-type=module', '-e', `
    import { readFile } from 'node:fs/promises';
    import { compileAllocation } from '@increment/assignment';
    import { runCases } from './run_cases.mjs';
    const vectors = JSON.parse(await readFile(process.argv[1], 'utf8'));
    console.log(JSON.stringify(await runCases(compileAllocation, vectors)));
  `, vectorsPath], consumer, true));
  assertCases(vectors, nodeResults, 'Node');
  run(process.execPath, [join(here, 'node_modules/typescript/bin/tsc'), '--noEmit', '--strict',
    '--module', 'NodeNext', '--target', 'ES2022', 'consumer.mts'], consumer);
  console.log(`${packed.name} ${packed.version}, Node ${process.version}: ${vectors.cases.length} cases and installed TypeScript declarations`);

  for (const [runtime, args] of [
    [python, ['-I', 'examples/server.py']],
    [process.execPath, ['examples/server.mjs']],
  ]) {
    assertExample(JSON.parse(run(runtime, args, consumer, true)), vectors);
  }
  console.log('Installed Python and Node quickstarts: membership, decisions, baseline/variant responses and enrollment ramp');
  await qualifyBrowser(consumer, vectors);
  await qualifyWorker(consumer, packageDir, vectors);
  console.log(`Qualified ${wheel} and ${packed.filename} outside the checkout`);
} finally {
  await rm(scratch, { recursive: true, force: true });
}
