import { chromium } from 'playwright';
const B = 'http://localhost:4321/build-your-own-3dgs';
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));

// en 第1章 (翻訳済み)
await page.goto(`${B}/en/chapters/01/`, { waitUntil: 'domcontentloaded' });
console.log('en ch1 title:', await page.locator('h1').first().textContent());
console.log('en ch1 banner:', await page.locator('.translation-banner').count(), '(0が正)');
console.log('en ch1 lang:', await page.evaluate(() => document.documentElement.lang));
await page.screenshot({ path: 'screenshots/en-chapter-01.png' });

// en 第2章 (フォールバック)
await page.goto(`${B}/en/chapters/02/`, { waitUntil: 'domcontentloaded' });
console.log('en ch2 banner:', await page.locator('.translation-banner').count(), '(1が正)');
const canonical = await page.locator('link[rel=canonical]').getAttribute('href');
console.log('en ch2 canonical:', canonical);
await page.screenshot({ path: 'screenshots/en-chapter-02-fallback.png' });

// 言語切替リンク
const switchHref = await page.locator('.lang-switch').getAttribute('href');
console.log('en ch2 → ja link:', switchHref, switchHref === '/build-your-own-3dgs/chapters/02/' ? 'OK' : 'NG');
await page.goto(`${B}/chapters/03/`, { waitUntil: 'domcontentloaded' });
const sw2 = await page.locator('.lang-switch').getAttribute('href');
console.log('ja ch3 → en link:', sw2, sw2 === '/build-your-own-3dgs/en/chapters/03/' ? 'OK' : 'NG');

// en ランディング
await page.goto(`${B}/en/`, { waitUntil: 'networkidle' });
console.log('en landing h2:', await page.locator('.chapters h2').textContent());
await page.screenshot({ path: 'screenshots/en-landing.png', fullPage: false });

if (errors.length) { console.log('ERRORS:', errors); process.exitCode = 1; }
await browser.close();
