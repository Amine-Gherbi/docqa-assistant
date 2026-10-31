#!/usr/bin/env node
/**
 * Emit a static entry for every client route the app serves.
 *
 * The app is a single-page Vite build published to a static host with no
 * rewrite/fallback rule, so a direct request for a client route (`/eval`) hits
 * the file system and 404s. Copying the built `index.html` to
 * `dist/<route>/index.html` lets the host answer the route with the same entry
 * document; its asset URLs are root-absolute, so they still resolve.
 *
 * The route list is derived from two independent places in App.tsx: the router
 * predicate (`isEvalRoute`) and the navigation links that expose each route.
 * The build fails when they disagree, so neither can drift silently.
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const UI_ROOT = resolve(HERE, '..');

function uniqueSorted(values) {
  return [...new Set(values)].sort();
}

function normaliseRoute(route) {
  const path = route.split(/[?#]/, 1)[0];
  const trimmed = path.replace(/\/+$/, '');
  return trimmed === '' ? '/' : trimmed;
}

function routerRoutes(source) {
  const fn =
    /function\s+isEvalRoute\s*\([^)]*\)\s*(?::\s*[^{]+)?\{([\s\S]*?)\n\}/.exec(
      source,
    );
  if (!fn) {
    throw new Error('emit-routes: could not find isEvalRoute() in src/App.tsx');
  }

  const routes = [];
  for (const match of fn[1].matchAll(/\bpath\s*===\s*['"]([^'"]+)['"]/g)) {
    routes.push(normaliseRoute(match[1]));
  }
  return uniqueSorted(routes);
}

function linkedRoutes(source) {
  const nav = /<nav\b[\s\S]*?<\/nav>/.exec(source);
  if (!nav) {
    throw new Error(
      'emit-routes: could not find the primary <nav> in src/App.tsx',
    );
  }

  const routes = [];
  for (const match of nav[0].matchAll(/\bhref\s*=\s*["']([^"']+)["']/g)) {
    if (!match[1].startsWith('/')) {
      continue;
    }
    const route = normaliseRoute(match[1]);
    if (route !== '/') {
      routes.push(route);
    }
  }
  return uniqueSorted(routes);
}

export function deriveRoutes(source) {
  const router = routerRoutes(source);
  const linked = linkedRoutes(source);
  const disagree =
    router.length !== linked.length ||
    router.some((route, index) => route !== linked[index]);
  if (disagree) {
    throw new Error(
      `emit-routes: the navigation links [${linked.join(', ')}] and the ` +
        `router [${router.join(', ')}] disagree; add the route to both or ` +
        'neither.',
    );
  }
  if (router.length === 0) {
    throw new Error('emit-routes: no client routes were derived from src/App.tsx');
  }

  return { routes: router, routerRoutes: router, linkedRoutes: linked };
}

export function emitRouteEntries(distDir, routes) {
  const indexPath = join(distDir, 'index.html');
  let index;
  try {
    index = readFileSync(indexPath);
  } catch (error) {
    throw new Error(`emit-routes: cannot read ${indexPath}: ${error.message}`, {
      cause: error,
    });
  }

  const entry = /<script\b[^>]*\bsrc\s*=\s*["']([^"']+)["']/i.exec(
    index.toString('utf8'),
  );
  if (!entry || !entry[1].startsWith('/')) {
    throw new Error(
      'emit-routes: the built entry script is not root-absolute; ' +
        'set Vite base to "/" so a copy under a route directory still resolves.',
    );
  }

  const written = [];
  for (const route of routes) {
    const segment = route.replace(/^\/+|\/+$/g, '');
    if (
      segment === '' ||
      segment.split('/').some((part) => part === '.' || part === '..')
    ) {
      throw new Error(`emit-routes: refusing unsafe route "${route}"`);
    }
    const target = join(distDir, segment, 'index.html');
    mkdirSync(dirname(target), { recursive: true });
    // A Buffer copy keeps the document byte-identical, so the hashed asset
    // URLs and module preload match the entry chunk exactly.
    writeFileSync(target, index);
    written.push(target);
  }
  return written;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const source = readFileSync(join(UI_ROOT, 'src', 'App.tsx'), 'utf8');
  const { routes } = deriveRoutes(source);
  const dist = process.argv[2]
    ? resolve(process.argv[2])
    : join(UI_ROOT, 'dist');
  const written = emitRouteEntries(dist, routes);
  console.log(
    `emit-routes: ${routes.join(', ')} → ` +
      written.map((path) => relative(UI_ROOT, path)).join(', '),
  );
}
