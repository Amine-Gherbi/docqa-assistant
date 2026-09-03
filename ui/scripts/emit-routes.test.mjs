import {
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

import { deriveRoutes, emitRouteEntries } from './emit-routes.mjs';

const appSource = readFileSync(join(process.cwd(), 'src', 'App.tsx'), 'utf8');

function routeSource({ router, links }) {
  const branches = router
    .map((route) => `  if (path === '${route}') return true;`)
    .join('\n');
  const anchors = links
    .map((route) => `      <a href="${route}">${route}</a>`)
    .join('\n');
  return [
    'function isEvalRoute(): boolean {',
    "  const path = window.location.pathname.replace(/\\/+$/, '');",
    branches,
    '  return false;',
    '}',
    'function SiteHeader() {',
    '  return (',
    '    <nav className="site-nav">',
    anchors,
    '    </nav>',
    '  );',
    '}',
  ].join('\n');
}

describe('deriveRoutes', () => {
  it('derives exactly the client routes the app serves', () => {
    const { routes, routerRoutes, linkedRoutes } = deriveRoutes(appSource);

    expect(routes).toEqual(['/eval']);
    expect(routerRoutes).toEqual(['/eval']);
    expect(linkedRoutes).toEqual(['/eval']);
  });

  it('fails when a served link has no matching router branch', () => {
    const source = routeSource({
      router: ['/eval'],
      links: ['/eval', '/report'],
    });

    expect(() => deriveRoutes(source)).toThrow(/disagree/i);
  });

  it('fails when the router serves a route no link reaches', () => {
    const source = routeSource({
      router: ['/eval', '/report'],
      links: ['/eval'],
    });

    expect(() => deriveRoutes(source)).toThrow(/disagree/i);
  });

  it('fails when the router predicate cannot be found', () => {
    expect(() => deriveRoutes('<nav><a href="/eval">Eval</a></nav>')).toThrow(
      /isEvalRoute/,
    );
  });
});

describe('emitRouteEntries', () => {
  let dist;

  afterEach(() => {
    rmSync(dist, { recursive: true, force: true });
  });

  it('copies the built index byte-for-byte into each route directory', () => {
    dist = mkdtempSync(join(tmpdir(), 'docqa-routes-'));
    const index = '<script type="module" crossorigin src="/assets/index-abc.js"></script>';
    writeFileSync(join(dist, 'index.html'), index);

    const written = emitRouteEntries(dist, ['/eval']);

    expect(written).toEqual([join(dist, 'eval', 'index.html')]);
    expect(readFileSync(written[0])).toEqual(
      readFileSync(join(dist, 'index.html')),
    );
  });

  it('refuses a build whose entry script is not root-absolute', () => {
    dist = mkdtempSync(join(tmpdir(), 'docqa-routes-'));
    mkdirSync(dist, { recursive: true });
    writeFileSync(
      join(dist, 'index.html'),
      '<script type="module" crossorigin src="./assets/index-abc.js"></script>',
    );

    expect(() => emitRouteEntries(dist, ['/eval'])).toThrow(/absolute/i);
  });
});
