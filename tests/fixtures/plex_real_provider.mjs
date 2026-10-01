// Private synthetic provider execution. No credentials, real clients or network.
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { stripTypeScriptTypes } from 'node:module';

if (!process.permission || process.permission.has('net') || process.permission.has('fs.write')) {
  throw new Error('Synthetic provider requires network/write-denied Node permissions');
}
const sourcePath = process.argv[2];
const source = readFileSync(sourcePath, 'utf8');
const expected = 'b2a6d66db3dfe77e7ca1cc67e76f0e8620fd19ce053baf5bc80ad1a2cd58e35a';
if (createHash('sha256').update(source).digest('hex') !== expected) {
  throw new Error('Pinned Plex provider source mismatch');
}
globalThis.fetch = () => { throw new Error('Network forbidden in synthetic fixture'); };
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const input = JSON.parse(raw);
if (input.call.name !== 'plugin__plex__plex_list_library') {
  throw new Error('Unsupported fixture tool');
}
const violations = [];
function client(method, rows) {
  return new Proxy(Object.freeze({
    [method]: async () => structuredClone(rows),
  }), {
    get(target, key) {
      if (Object.hasOwn(target, key)) return target[key];
      violations.push(String(key));
      throw new Error(`Unexpected synthetic client method: ${String(key)}`);
    },
  });
}
// All imports in the pinned source are type-only. Stripping and importing a
// data URL avoids loading its package or constructing any real API clients.
const code = stripTypeScriptTypes(source, { mode: 'strip' });
const { buildPlexTools } = await import('data:text/javascript;base64,' + Buffer.from(code).toString('base64'));
const tools = buildPlexTools({
  radarr: client('getMovies', input.movies),
  sonarr: client('getSeries', input.series),
});
const tool = tools.find(t => t.name === 'plex_list_library');
if (!tool) throw new Error('Pinned tool missing');
const response = await tool.execute(input.call.arguments);
if (violations.length) throw new Error(`Forbidden client access: ${violations.join(',')}`);
process.stdout.write(JSON.stringify(response));
