const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync('acap/app/html/app.js', 'utf8');
const functions = source.slice(source.indexOf('  function fuseStream('), source.indexOf('  function currentCfg('));
const { simulate, fuseStream } = vm.runInNewContext(`
  const HOP_S = .48;
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const thrOf = (s) => .95 - .009 * (clamp(+s, 1, 100) - 1);
  const hornThrOf = (s) => .6 - .0055 * (clamp(+s, 1, 100) - 1);
  ${functions}
  ({simulate, fuseStream});
`);
const cfg = { sens: 10, hornSens: 50, hornTriggers: false, bellTriggers: true,
  smooth: 2, trigger: 3, release: 8, cool: 10, maxEvent: 600, minLevel: -90 };
const cases = JSON.parse(fs.readFileSync('tests/fixtures/runtime-cases.json', 'utf8'));
for (const c of cases) {
  const points = c.records.map((r, j) => ({ t: j * .48, raw: 0, horn: 0, trainBell: r.bell, level: r.level }));
  assert.equal(simulate(points, cfg).length > 0, c.expected_bell_event, c.name);
  assert.equal(simulate(points, { ...cfg, bellTriggers: false }).length, 0, `${c.name}: toggle off`);
}
const horns = [0, 1, 2].map(j => ({ t: j * .48, head: 0, level: -30, g: [0, 0, .99, 0, 0, 0] }));
const fused = fuseStream(horns, { ...cfg, otherHorns: true, source: 'head', speechSup: 0, requireRail: false, minRise: 0 });
assert.equal(simulate(fused, { ...cfg, hornTriggers: true }).length, 1, 'other horns match unfiltered runtime without rail context');
console.log('UI preview: real model bell cases, toggle off and other-horn parity passed');

