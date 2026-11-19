import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { compileAllocation } from '@increment/assignment';

const here = dirname(fileURLToPath(import.meta.url));
const read = (file) => readFile(join(here, file), 'utf8');

async function quickstart() {
  const initial = compileAllocation(JSON.parse(await read('config.json')));
  const ramped = compileAllocation(JSON.parse(await read('config-ramped.json')));
  const requests = [
    ['acct:1', 'checkout'], ['acct:2', 'checkout'], ['acct:3', 'checkout'],
    ['acct:1', 'search'], ['acct:1', 'e'],
  ];

  // Log only the first membership per unit in the shopping study, before treatment.
  const memberships = [];
  for (const unitId of new Set(requests.map(([unitId]) => unitId))) {
    memberships.push({ unit_id: unitId, membership: await initial.holdoutMembership('shopping', unitId) });
  }
  const assignments = [];
  const responses = [];
  for (const [unitId, target] of requests) {
    const decision = await initial.decide(target, unitId);
    assignments.push({ unit_id: unitId, decision });
    // A response body is not proof the client rendered it.
    const content = decision.arm === 'treatment' ? 'New experience' : 'Baseline experience';
    responses.push({ unit_id: unitId, target, content });
  }
  return {
    memberships, assignments, responses,
    ramp: {
      before: await initial.decide('checkout', 'acct:1'),
      after: await ramped.decide('checkout', 'acct:1'),
    },
  };
}

export async function serve() {
  const html = await read('index.html');
  const config = await read('config.json');
  const client = await read('client.mjs');
  const evaluator = compileAllocation(JSON.parse(config));
  const sdk = dirname(fileURLToPath(import.meta.resolve('@increment/assignment')));
  const reply = (res, status, type, body) => res.writeHead(status, { 'Content-Type': type }).end(body);

  const server = createServer(async (req, res) => {
    try {
      const path = new URL(req.url, 'http://localhost').pathname;
      if (path === '/server' || path === '/client' || path === '/') {
        // In an application, resolve this identity from the incoming request.
        const selected = path === '/client' ? null : await evaluator.decide('checkout', 'acct:2');
        const safeJson = JSON.stringify(selected).replaceAll('<', '\\u003c');
        reply(res, 200, 'text/html; charset=utf-8', html.replace('<!-- decision -->', () => safeJson));
      } else if (path === '/client.mjs') {
        reply(res, 200, 'text/javascript; charset=utf-8', client);
      } else if (path === '/config.json') {
        reply(res, 200, 'application/json', config);
      } else if (/^\/sdk\/[\w-]+\.js$/.test(path)) {
        try {
          reply(res, 200, 'text/javascript; charset=utf-8', await readFile(join(sdk, path.slice(5))));
        } catch (error) {
          if (error.code !== 'ENOENT') throw error;
          reply(res, 404, 'text/plain', 'Not found');
        }
      } else {
        reply(res, 404, 'text/plain', 'Not found');
      }
    } catch {
      reply(res, 500, 'text/plain', 'Decision unavailable');
    }
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return {
    url: `http://127.0.0.1:${server.address().port}`,
    close: () => new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve())),
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (process.argv[2] === '--serve') {
    const { url } = await serve();
    console.log(`Server decision: ${url}/server\nBrowser decision: ${url}/client`);
  } else {
    console.log(JSON.stringify(await quickstart(), null, 2));
  }
}
