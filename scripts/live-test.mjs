import { chromium } from 'playwright';
const B = 'https://kogum4.github.io/build-your-own-3dgs';
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));

await page.goto(`${B}/chapters/01/`, { waitUntil: 'domcontentloaded' });
await page.waitForSelector('.exec-widget', { timeout: 20000 });
console.log('✓ 本番: execブロックがエディタ化');

// 実行テスト (Pyodide CDNロード込み)
const block = page.locator('.exec-widget').nth(1);
await block.scrollIntoViewIfNeeded();
await block.locator('.exec-run').click();
await block.locator('.out-stdout').waitFor({ timeout: 180000 });
const text = await block.locator('.out-stdout').textContent();
console.log(text?.includes('[[100.') ? '✓ 本番: Pyodide実行成功' : `✗ 出力異常: ${text}`);

// ウィジェット
const w = page.locator('[data-widget="ch1-covariance"]');
await w.scrollIntoViewIfNeeded();
await page.waitForTimeout(800);
console.log((await w.evaluate((el) => el.classList.contains('widget-ready'))) ? '✓ 本番: ウィジェットマウント' : '✗ ウィジェット未マウント');

// ch9
await page.goto(`${B}/chapters/09/`, { waitUntil: 'domcontentloaded' });
const w9 = page.locator('[data-widget="ch9-garden"]');
await w9.scrollIntoViewIfNeeded();
await page.waitForTimeout(2500);
const perf = await w9.locator('p').filter({ hasText: /ガウシアンを描画/ }).textContent().catch(() => null);
console.log(perf ? `✓ 本番: ch9レンダリング (${perf.trim()})` : '✗ ch9レンダリング失敗');

await page.screenshot({ path: 'screenshots/live-final.png' });
const filtered = errors.filter((e) => !e.includes('favicon'));
console.log(filtered.length ? `✗ エラー: ${filtered.join(' / ')}` : '✓ 本番: コンソールエラーなし');
if (filtered.length) process.exitCode = 1;
await browser.close();
