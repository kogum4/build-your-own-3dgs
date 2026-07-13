import { chromium } from 'playwright';
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));

// ---- ch5 ----
await page.goto('http://localhost:4321/build-your-own-3dgs/chapters/05/', { waitUntil: 'domcontentloaded' });
const w5 = page.locator('[data-widget="ch5-live-training"]');
await w5.scrollIntoViewIfNeeded();
await page.waitForTimeout(1500);
console.log('ch5 mounted:', await w5.evaluate((el) => el.classList.contains('widget-ready')));
// 録画再生の確認
const status1 = await w5.locator('p').filter({ hasText: /録画済み|Playing/ }).count();
console.log('ch5 replay playing:', status1 > 0);
await w5.screenshot({ path: 'screenshots/widget-ch5-replay.png' });
// ライブ学習開始 (Pyodideブート込み)
await w5.getByRole('button', { name: /自分のブラウザで学習する/ }).click();
try {
  await page.waitForFunction(
    () => document.querySelector('[data-widget="ch5-live-training"]')?.textContent?.match(/学習中|学習完了/),
    null, { timeout: 120000 },
  );
  console.log('ch5 live training started');
  // 完了 or 60秒進行を待つ
  await page.waitForFunction(
    () => document.querySelector('[data-widget="ch5-live-training"]')?.textContent?.includes('学習完了'),
    null, { timeout: 180000 },
  ).then(() => console.log('ch5 live training COMPLETED')).catch(() => console.log('ch5 live training still running (timeout probe)'));
} catch { console.log('ch5 live training FAILED to start'); process.exitCode = 1; }
await w5.screenshot({ path: 'screenshots/widget-ch5-live.png' });

// ---- ch9 ----
await page.goto('http://localhost:4321/build-your-own-3dgs/chapters/09/', { waitUntil: 'domcontentloaded' });
const w9 = page.locator('[data-widget="ch9-garden"]');
await w9.scrollIntoViewIfNeeded();
await page.waitForTimeout(2000);
console.log('ch9 mounted:', await w9.evaluate((el) => el.classList.contains('widget-ready')));
const perf = await w9.locator('p').filter({ hasText: /ガウシアンを描画|Rendered/ }).textContent().catch(() => null);
console.log('ch9 render perf:', perf?.trim());
// スライダーを動かす
const slider = w9.locator('input[type=range]');
await slider.fill('25');
await page.waitForTimeout(400);
// test視点比較
await w9.getByRole('button', { name: '3', exact: true }).click();
await page.waitForTimeout(500);
await w9.screenshot({ path: 'screenshots/widget-ch9-garden.png' });

if (errors.length) { console.log('PAGE ERRORS:', errors); process.exitCode = 1; }
await browser.close();
