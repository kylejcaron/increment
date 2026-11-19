import { readdirSync, readFileSync } from 'node:fs';
import { isBuiltin } from 'node:module';
import ts from 'typescript';

const source = new URL('../src/', import.meta.url);
for (const file of readdirSync(source, { recursive: true })) {
  if (!file.endsWith('.ts')) continue;
  const text = readFileSync(new URL(file, source), 'utf8');
  const { importedFiles } = ts.preProcessFile(text, true, true);
  for (const { fileName } of importedFiles) {
    if (fileName.startsWith('node:') || isBuiltin(fileName)) {
      console.error(`src/${file}: runtime sources cannot import ${fileName}`);
      process.exitCode = 1;
    }
  }
}
