#!/usr/bin/env node
// 実ブラウザ (システムの Edge) での動作検証。
// 前提: `pnpm preview` が http://localhost:4321 で起動していること。
//   node scripts/smoke-test.mjs [--screenshot-only]
import { chromium } from 'playwright';
import fs from 'node:fs';

const BASE = 'http://localhost:4321/build-your-own-3dgs';
const screenshotOnly = process.argv.includes('--screenshot-only');
const outDir = 'screenshots';
fs.mkdirSync(outDir, { recursive: true });

const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });

const errors = [];
page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
page.on('console', (msg) => {
  if (msg.type() === 'error') errors.push(`console.error: ${msg.text()}`);
});

function ok(label) {
  console.log(`  ✓ ${label}`);
}

function fail(label, detail) {
  console.error(`  ✗ ${label}${detail ? `: ${detail}` : ''}`);
  process.exitCode = 1;
}

// ---------- ランディング ----------
console.log('== ランディング ==');
await page.goto(`${BASE}/`, { waitUntil: 'networkidle' });
await page.screenshot({ path: `${outDir}/landing.png`, fullPage: true });
const cardCount = await page.locator('.chapter-card').count();
cardCount === 16 ? ok('章カード16枚') : fail('章カード16枚', `actual=${cardCount}`);

// ---------- 第1章 ----------
console.log('== 第1章 ==');
await page.goto(`${BASE}/chapters/01/`, { waitUntil: 'domcontentloaded' });
await page.waitForSelector('.exec-widget', { timeout: 15000 });
await page.screenshot({ path: `${outDir}/chapter-01-top.png` });
ok('exec ブロックがエディタ化された');

const widgets = await page.locator('.exec-widget').count();
console.log(`  exec-widget: ${widgets}個`);

if (!screenshotOnly) {
  // 2番目のブロック (共分散行列を print する動作確認ブロック) を実行
  const block = page.locator('.exec-widget').nth(1);
  await block.scrollIntoViewIfNeeded();
  await block.locator('.exec-run').click();

  // Pyodide 初回ロード込みで待つ
  const stdout = block.locator('.out-stdout');
  try {
    await stdout.waitFor({ timeout: 180000 });
    const text = await stdout.textContent();
    text?.includes('[[100.')
      ? ok(`実行結果 OK: ${text.trim().split('\n')[0]}`)
      : fail('実行結果の内容', text ?? '(empty)');
  } catch {
    fail('実行結果が出力されない (timeout)');
  }
  await block.screenshot({ path: `${outDir}/chapter-01-exec.png` });

  // 期待出力が折りたたまれたか
  const details = block.locator('xpath=following-sibling::details[1]');
  if ((await details.count()) > 0) {
    (await details.getAttribute('open')) === null
      ? ok('期待出力が折りたたまれた')
      : fail('期待出力の折りたたみ');
  }

  // matplotlib + 日本語フォント (エディタを書き換えて図を出す)
  const first = page.locator('.exec-widget').nth(0);
  await first.scrollIntoViewIfNeeded();
  await first.locator('.cm-content').click();
  await page.keyboard.press('Control+a');
  await page.keyboard.insertText(
    [
      'import matplotlib.pyplot as plt',
      'plt.figure(figsize=(4,2.4))',
      'plt.plot([0,1,2,3],[0,1,4,9])',
      'plt.title("日本語タイトルの確認")',
      'print("plotted")',
    ].join('\n'),
  );
  await first.locator('.exec-run').click();
  try {
    await first.locator('.out-figure').waitFor({ timeout: 120000 });
    ok('matplotlib 図が表示された');
    await first.screenshot({ path: `${outDir}/chapter-01-matplotlib.png` });
  } catch {
    fail('matplotlib 図が表示されない (timeout)');
    await first.screenshot({ path: `${outDir}/chapter-01-matplotlib-fail.png` });
  }

  // ステータスピル
  const pill = await page.locator('.py-status-pill').textContent();
  console.log(`  status pill: ${pill?.trim()}`);
}

// ---------- コンソールエラー ----------
const filtered = errors.filter(
  (e) => !e.includes('favicon') && !e.includes('net::ERR_') && !e.includes('404'),
);
if (filtered.length > 0) {
  console.log('== ブラウザエラー ==');
  filtered.forEach((e) => console.log(`  ${e}`));
  process.exitCode = 1;
} else {
  ok('コンソールエラーなし');
}

await browser.close();
