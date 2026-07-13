import { chromium } from 'playwright';
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.goto('http://localhost:4321/build-your-own-3dgs/chapters/01/', { waitUntil: 'domcontentloaded' });
// 各ウィジェットへスクロールしてマウントを促す
for (const name of ['ch1-covariance', 'ch1-rgb-mixer']) {
  const w = page.locator(`[data-widget="${name}"]`);
  await w.scrollIntoViewIfNeeded();
  await page.waitForTimeout(600);
  const ready = await w.evaluate((el) => el.classList.contains('widget-ready'));
  console.log(`${name}: ${ready ? 'mounted' : 'NOT mounted'}`);
  await w.screenshot({ path: `screenshots/widget-${name}.png` });
}
// スライダー操作で再描画されるか (rgb-mixer)
const slider = page.locator('[data-widget="ch1-rgb-mixer"] input[type=range]').first();
await slider.fill('30');
await page.waitForTimeout(300);
await page.locator('[data-widget="ch1-rgb-mixer"]').screenshot({ path: 'screenshots/widget-rgb-after.png' });
console.log('slider interaction OK');
if (errors.length) { console.log('ERRORS:', errors); process.exitCode = 1; }
await browser.close();
