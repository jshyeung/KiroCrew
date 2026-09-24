/**
 * Screenshot probe: the sidebar's folder sort mode (Custom / Name / Created).
 *
 * Runs the REAL built SPA (website/dist) behind a tiny in-process static server
 * and answers every /api/** call from fixtures via Playwright route interception
 * (gateway-free - no kiro-cli, no live backend). The fixture is the reporter's
 * own scheme from #13321: folders numbered 01. .. 04., 98., 99. whose STORED
 * positions were set by placing them, so the custom order draws `99.` between
 * `03.` and `04.` and `98.` below - exactly his screenshot.
 *
 * The mode is driven through the real control: the sort-and-filter menu's
 * "Folder order" rows. A PATCH to /api/config/kirocrew is answered by mutating the
 * fixture the GET serves, the way the real endpoint pair round-trips, so the
 * settle-time refetch sees the saved value.
 *
 * Every frame is asserted before it is written: the rendered folder rows must be
 * in the order the mode promises, or the run fails instead of shipping a picture
 * of the wrong state.
 *
 * Frames written:
 *   01-custom-before   the stored order (today's behaviour and the default)
 *   02-menu-custom     the menu with "Folder order" -> Custom checked
 *   03-name-after      the same folders after picking Name: 01 < 02 < ... < 99
 *   04-menu-name       the menu with Name checked
 *   05-created-after   after picking Created (Newest): newest first
 *
 * Usage: node scripts/capture-folder-sort-mode.mjs [outDir]
 */
import { chromium } from 'playwright'
import { mkdirSync } from 'node:fs'
import { serveDist } from './lib/serve-dist.mjs'
import { KIROCREW_CONFIG_FIXTURE, json, logPageProblems, stubDashboardApi } from './lib/stub-dashboard-api.mjs'

const OUT = process.argv[2] || '../temp-screenshots/folder-sort-mode'
mkdirSync(OUT, { recursive: true })

// The reporter's sidebar: stored positions from the drags that placed them.
// `created_at` stamps are the order the folders were made in (98 and 99 last).
const folders = [
  { id: 'f01', name: '01. Inbox', order: 0, collapsed: true, created_at: 1_758_600_000 },
  { id: 'f02', name: '02. Research', order: 1, collapsed: true, created_at: 1_758_610_000 },
  { id: 'f03', name: '03. Writing', order: 2, collapsed: true, created_at: 1_758_620_000 },
  { id: 'f99', name: '99. Archive', order: 3, collapsed: true, created_at: 1_758_650_000 },
  { id: 'f04', name: '04. Reviews', order: 4, collapsed: true, created_at: 1_758_630_000 },
  { id: 'f98', name: '98. Someday', order: 5, collapsed: true, created_at: 1_758_640_000 },
]
const slots = [{
  key: 's1', title: 'Weekly planning', messages: 4, running: false, agent: 'kirocrew',
  created: '2026-09-23T09:00:00Z', last_ts: '2026-09-23T09:30:00Z', folder_id: 'f01',
}]

const CUSTOM = ['f01', 'f02', 'f03', 'f99', 'f04', 'f98']
const BY_NAME = ['f01', 'f02', 'f03', 'f04', 'f98', 'f99']
const NEWEST_FIRST = ['f99', 'f98', 'f04', 'f03', 'f02', 'f01']

/** The fake gateway's stored mode; PATCH writes it, GET serves it. */
let storedMode = 'custom'

async function drawnOrder(page) {
  return page.$$eval('[data-folder-row]', els => els.map(el => el.getAttribute('data-folder-row')))
}

async function assertOrder(page, expected, label) {
  await page.waitForFunction(
    exp => JSON.stringify([...document.querySelectorAll('[data-folder-row]')].map(el => el.getAttribute('data-folder-row'))) === JSON.stringify(exp),
    expected,
    { timeout: 8000 },
  ).catch(() => {})
  const got = await drawnOrder(page)
  if (JSON.stringify(got) !== JSON.stringify(expected)) {
    throw new Error(`${label}: rendered order ${JSON.stringify(got)} != expected ${JSON.stringify(expected)}`)
  }
  console.log(`${label}: rendered order OK ${got.join(' ')}`)
}

async function openMenu(page) {
  await page.getByRole('button', { name: 'Sort and filter sessions' }).click()
  await page.locator('[data-testid="folder-order-custom"]').waitFor({ state: 'visible', timeout: 5000 })
  await page.waitForTimeout(250)
}

async function sidebarClip(page) {
  const row = page.locator('[data-folder-row="f01"]')
  const box = await row.first().boundingBox()
  const x = box ? Math.max(0, box.x - 28) : 0
  return { x, y: 64, width: Math.min(420, 1100 - x), height: 300 }
}

async function main() {
  const served = await serveDist()
  const browser = await chromium.launch()
  const context = await browser.newContext({ viewport: { width: 1100, height: 760 }, deviceScaleFactor: 2 })
  const page = await context.newPage()
  await stubDashboardApi(page, {
    folders, slots,
    extra: async (path, route) => {
      if (path !== '/api/config/kirocrew') return false
      const method = route.request().method()
      if (method === 'PATCH') {
        const body = JSON.parse(route.request().postData() || '{}')
        if (body.path === 'dashboard.folder_sort') storedMode = body.value
        console.log(`STUB: PATCH ${body.path} = ${body.value}`)
        await json(route, { ok: true })
        return true
      }
      await json(route, { ...KIROCREW_CONFIG_FIXTURE, dashboard: { ...(KIROCREW_CONFIG_FIXTURE.dashboard || {}), folder_sort: storedMode } })
      return true
    },
  })
  logPageProblems(page)
  await page.goto(served.base + '/chat', { waitUntil: 'domcontentloaded' })
  await page.waitForTimeout(2600)

  // 01: the stored order, as today's build draws it (Custom is the default).
  await assertOrder(page, CUSTOM, 'custom (before)')
  await page.screenshot({ path: `${OUT}/01-custom-before.png`, clip: await sidebarClip(page) })
  console.log('wrote', `${OUT}/01-custom-before.png`)

  // 02: the control, Custom checked.
  await openMenu(page)
  await page.screenshot({ path: `${OUT}/02-menu-custom.png`, clip: { x: 150, y: 60, width: 560, height: 700 } })
  console.log('wrote', `${OUT}/02-menu-custom.png`)

  // 03: pick Name -> PATCH -> the tree re-sorts, positions untouched.
  await page.locator('[data-testid="folder-order-name"]').click()
  await page.keyboard.press('Escape')
  await assertOrder(page, BY_NAME, 'name (after)')
  if (storedMode !== 'name') throw new Error(`the menu did not PATCH dashboard.folder_sort=name (stored: ${storedMode})`)
  await page.screenshot({ path: `${OUT}/03-name-after.png`, clip: await sidebarClip(page) })
  console.log('wrote', `${OUT}/03-name-after.png`)

  // 04: the control again, Name checked.
  await openMenu(page)
  await page.screenshot({ path: `${OUT}/04-menu-name.png`, clip: { x: 150, y: 60, width: 560, height: 700 } })
  console.log('wrote', `${OUT}/04-menu-name.png`)

  // 05: Created (Newest).
  await page.locator('[data-testid="folder-order-created"]').click()
  await page.keyboard.press('Escape')
  await assertOrder(page, NEWEST_FIRST, 'created (after)')
  if (storedMode !== 'created') throw new Error(`the menu did not PATCH dashboard.folder_sort=created (stored: ${storedMode})`)
  await page.screenshot({ path: `${OUT}/05-created-after.png`, clip: await sidebarClip(page) })
  console.log('wrote', `${OUT}/05-created-after.png`)

  // And back to Custom: the stored order returns exactly, since nothing rewrote it.
  await openMenu(page)
  await page.locator('[data-testid="folder-order-custom"]').click()
  await page.keyboard.press('Escape')
  await assertOrder(page, CUSTOM, 'custom (restored)')

  await context.close()
  await browser.close()
  served.srv.close()
}

main().catch(err => { console.error(err); process.exit(1) })
