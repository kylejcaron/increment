import { readFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { DuckDBConnection } from '@malloydata/db-duckdb';
import { SingleConnectionRuntime } from '@malloydata/malloy';

function identifier(name) {
  if (typeof name !== 'string' || !/^[A-Za-z_][A-Za-z_0-9]*$/.test(name)) {
    throw new Error(`Invalid SQL identifier: ${JSON.stringify(name)}`);
  }
  return `"${name}"`;
}

// The caller supplies every (table, row_key) from the selected catalog closure.
// Use the very same connection for the checks, model compilation and queries.
async function invalidKey(connection, keys) {
  for (const [relation, key] of keys) {
    const table = relation.split('.').map(identifier).join('.');
    const column = identifier(key);
    const { rows: [row] } = await connection.runSQL(`
      SELECT count(*) <> count(${column}) AS has_null,
             count(${column}) <> count(DISTINCT ${column}) AS has_duplicate
      FROM ${table}`);
    if (row.has_null || row.has_duplicate) {
      return { code: row.has_null ? 'invalid_key.null' : 'invalid_key.duplicate', relation, key };
    }
  }
}

function jsonValue(_key, value) {
  if (typeof value !== 'bigint') return value;
  const number = Number(value);
  return Number.isSafeInteger(number) ? number : value.toString();
}

async function run({ database, model, keys, queries }) {
  const connection = new DuckDBConnection({
    name: 'duckdb',
    databasePath: resolve(database),
    readOnly: true,
    shareable: false,
    enableExternalAccess: false,
  });
  try {
    const invalid = await invalidKey(connection, keys);
    if (invalid) {
      console.error(JSON.stringify(invalid));
      process.exitCode = 1;
      return;
    }
    const runtime = new SingleConnectionRuntime({
      connection,
      urlReader: { readURL: url => readFile(url, 'utf8') },
    });
    const report = runtime.loadModel(pathToFileURL(resolve(model)));
    const rows = Object.create(null);
    for (const name of queries) {
      const result = await report.loadQueryByName(name).run();
      rows[name] = result.data.toObject();
    }
    console.log(JSON.stringify(rows, jsonValue));
  } finally {
    await connection.close();
  }
}

try {
  await run(JSON.parse(readFileSync(0, 'utf8')));
} catch (error) {
  console.error(JSON.stringify({
    code: 'malloy.query',
    message: error.message,
    problems: error.problems?.map(({ code, message }) => ({ code, message })),
  }));
  process.exitCode = 1;
}
