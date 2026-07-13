import { chromium } from 'playwright';
const targets = [
  ['02', 'ch2-alpha'], ['03', 'ch3-autograd-graph'], ['04', 'ch4-broadcast'],
  ['06', 'ch6-ellipsoid'], ['07', 'ch7-camera-projection'], ['08', 'ch8-ewa'],
];
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
for (const [ch, name] of targets) {
  await page.goto(`http://localhost:4321/build-your-own-3dgs/chapters/${ch}/`, { waitUntil: 'domcontentloaded' });
  const w = page.locator(`[data-widget="${name}"]`);
  if ((await w.count()) === 0) { console.log(`${name}: マウントポイントなし!`); process.exitCode = 1; continue; }
  await w.scrollIntoViewIfNeeded();
  await page.waitForTimeout(800);
  const ready = await w.evaluate((el) => el.classList.contains('widget-ready'));
  console.log(`${name}: ${ready ? 'mounted' : 'NOT mounted'}`);
  if (!ready) process.exitCode = 1;
  await w.screenshot({ path: `screenshots/widget-${name}.png` });
}
if (errors.length) { console.log('PAGE ERRORS:', errors); process.exitCode = 1; }
await browser.close();
