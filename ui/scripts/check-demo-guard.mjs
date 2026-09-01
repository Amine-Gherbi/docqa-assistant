#!/usr/bin/env node
/**
 * Refuse to ship the offline answer fixture in a production bundle.
 *
 * The fixture is a last-resort development aid; a hosted build must answer
 * through the API. `VITE_DEMO=1` is the only way to opt back in, and the
 * caller must state that choice in the environment.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const dist = process.argv[2] ?? 'dist';
const allowFixture = process.env.VITE_DEMO === '1';
const needles = [
  'fixtures/demo-answer',
  'Offline demo fixture used only when the API is unreachable',
];

function* walk(directory) {
  for (const name of readdirSync(directory)) {
    const path = join(directory, name);
    if (statSync(path).isDirectory()) {
      yield* walk(path);
    } else {
      yield path;
    }
  }
}

let files;
try {
  files = [...walk(dist)];
} catch (error) {
  console.error(`demo guard: cannot read ${dist}: ${error.message}`);
  process.exit(1);
}

const hits = [];
for (const file of files) {
  if (!/\.(js|mjs|css|html|json|map)$/.test(file)) {
    continue;
  }
  const text = readFileSync(file, 'utf8');
  const found = needles.filter((needle) => text.includes(needle));
  if (found.length > 0) {
    hits.push({ file: relative(dist, file), found });
  }
}

if (hits.length > 0 && !allowFixture) {
  console.error(
    'demo guard: the offline fixture is in the production bundle; ' +
      'set VITE_DEMO=1 only for a deliberate demo build.',
  );
  for (const hit of hits) {
    console.error(`  ${hit.file}: ${hit.found.join(', ')}`);
  }
  process.exit(1);
}

if (hits.length > 0) {
  console.log(`demo guard: fixture allowed by VITE_DEMO=1 (${hits.length} file(s))`);
} else {
  console.log('demo guard: no offline fixture in the production bundle');
}
