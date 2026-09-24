/**
 * Screenshot harness for the workspace tree's STATE ROW under a childless
 * folder (#13054): "Empty folder", "No visible files", "Folder not readable",
 * "Files not shown: file limit reached" -- and for the listing-failure notice
 * that replaces the endless shimmer when the first `/api/project/tree` request
 * fails.
 *
 * Same house pattern as `capture-pierre-files-tab.mjs`: the REAL built SPA
 * behind the shared in-process static server, every `/api/**` answered from
 * fixtures via Playwright route interception -- gateway-free. The client code
 * under test is unmodified. Run it once against a dist built from `main` and
 * once against the branch's dist; the fixture and the clicks are identical, so
 * the two frames differ only by what the tree paints under an expanded folder.
 *
 * Fixture: the reporter's shape. A NON-repository workspace whose `_bg/` holds
 * only a hidden `.kiro/` folder (`hiddenOnlyDirectories`), an `empty/` folder
 * that is empty on disk, a `big/` folder whose files fell to the file cap
 * (`truncatedDirectories`), a `vault/` folder whose only entry `locked/` the
 * server could not read (`unreadableDirectories` -- the folder is listed, what
 * is beneath it is qualified, and `vault/` itself is not called empty), and a
 * populated `src/`. A dist built before this fix ignores the qualifier lists
 * and paints nothing under any of them.
 *
 * Frames (`<prefix>` is the --prefix argument, e.g. `before` / `after`):
 *   <prefix>-10-folders-expanded   `_bg`, `empty`, `big`, `vault`, `vault/locked`
 *                                  and `src` expanded
 *   <prefix>-11-bg-collapsed       `_bg` collapsed again (its row must go)
 *   <prefix>-20-listing-failed     the tree request answers 503: the HOST's
 *                                  own notice + Refresh (not changed by this
 *                                  PR; captured to show the failure path is
 *                                  not silent)
 *
 * Every frame is gated: the surface it names must have demonstrably rendered
 * (shadow-piercing probes, then a bytes-per-pixel blank check).
 *
 * Usage: node scripts/capture-files-tree-empty-folder.mjs <outDir> --prefix <p> [--dist <path>] [--expect-state-rows]
 */
import { chromium } from 'playwright'
import { mkdirSync, readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { serveDist, DEFAULT_DIST } from './lib/serve-dist.mjs'
import { logPageProblems, stubDashboardApi, json } from './lib/stub-dashboard-api.mjs'
import { chromiumExecutable } from './lib/chromium-executable.mjs'

const args = process.argv.slice(2)
const flag = (name) => { const i = args.indexOf(name); return i === -1 ? null : args[i + 1] }
const FLAG_VALUES = new Set(['--prefix', '--dist'].map(f => flag(f)).filter(Boolean))
const OUT = args.find(a => !a.startsWith('--') && !FLAG_VALUES.has(a)) || '../temp-screenshots/files-tree-empty-folder'
const PREFIX = flag('--prefix') || 'after'
const DIST = flag('--dist') ? resolve(flag('--dist')) : DEFAULT_DIST
/** With the flag, the state rows MUST render (the fix is in the dist); without
 *  it they MUST NOT (a pre-fix dist) -- the BEFORE frame is only evidence if it
 *  proves the absence it claims. */
const EXPECT_STATE_ROWS = args.includes('--expect-state-rows')

const PROJECT = resolve(dirname(fileURLToPath(import.meta.url)), '../..')
const SLOT = 'chat-files-tree-empty'
const MAX_EDGE = 2000
const MIN_MBPP = 15

mkdirSync(OUT, { recursive: true })

// ── Fixtures ────────────────────────────────────────────────────────────────

const TREE = {
  root: PROJECT,
  repo: false,
  paths: ['HEARTBEAT.md', 'notes/todo.md', 'src/app.py', 'src/util.py'],
  directories: ['_bg', 'big', 'empty', 'notes', 'src', 'vault', 'vault/locked'],
  truncated: true,
  truncatedDirectories: ['big'],
  hiddenOnlyDirectories: ['_bg'],
  unreadableDirectories: ['vault/locked'],
}

const STATE_LABELS = {
  '_bg/': 'No visible files',
  'empty/': 'Empty folder',
  'big/': 'Files not shown: file limit reached',
  'vault/locked/': 'Folder not readable',
}
// Every state row's name ends in this zero-width space -- the marker the
// tree's stylesheet selects (`src/pierre/treeStateRows.ts`), so a row that
// carries it is proven to be the styled synthetic row and not a real file.
const STATE_ROW_MARKER = '\u200b'
const isStateRow = (row, label) => row.text === label + STATE_ROW_MARKER

const slots = [{
  key: SLOT,
  title: 'Files tree',
  running: false,
  last_message: 'Files tree',
  messages: 2,
  agent: 'kirocrew',
  memory_mode: 'persistent',
  project: PROJECT,
  modified: Math.floor(Date.now() / 1000),
  source_links: [],
  source_links_total: 0,
}]
const t0 = Math.floor(Date.now() / 1000) - 900
const slotDetail = {
  running: false, has_more: false, total: 2, queue: [],
  messages: [
    { role: 'user', content: 'Show me the workspace files.', ts: String(t0) },
    { role: 'assistant', content: 'The Files tab is open on the right.', ts: String(t0 + 30) },
  ],
}
const FILES_TAB = { id: 'files', kind: 'files', title: 'Files' }
const bucket = (tabs, activeId) => JSON.stringify({ activeId, tabs })

function pngSize(path) {
  const b = readFileSync(path)
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }
}

async function main() {
  console.log('dist:', DIST, ' prefix:', PREFIX, ' expect state rows:', EXPECT_STATE_ROWS)
  const { srv, base } = await serveDist(DIST)
  const executablePath = chromiumExecutable()
  const browser = await chromium.launch({ executablePath })
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 })
  const page = await context.newPage()

  let treeFails = false
  const extra = async (path, route) => {
    if (path === '/api/chat/slots') return json(route, slots), true
    if (/^\/api\/chat\/slots\/[^/]+/.test(path)) return json(route, slotDetail), true
    if (path === '/api/project/tree') {
      return treeFails
        ? json(route, { error: 'Couldn’t list the workspace.', code: 'project_tree_unavailable' }, 503)
        : json(route, TREE)
      , true
    }
    // Not a repository: the git probes say so and the status query never fires.
    if (path === '/api/project/git') return json(route, { path: PROJECT, repo: false }), true
    if (path === '/api/project/git/status') return json(route, { repo: false, files: [] }), true
    if (path === '/api/project/git/log') return json(route, { repo: false, commits: [] }), true
    if (path === '/api/recent-projects') return json(route, { dirs: [PROJECT] }), true
    return false
  }
  await stubDashboardApi(page, { slots, extra })
  logPageProblems(page)

  const wrote = []
  function record(file, evidence) {
    const { w, h } = pngSize(file)
    const bytes = readFileSync(file).length
    const mbpp = Math.round((bytes * 1000) / (w * h))
    const over = w > MAX_EDGE || h > MAX_EDGE
    const blank = mbpp < MIN_MBPP
    console.log(`wrote ${file}  ${w}x${h}  ${bytes}B  ${mbpp} milli-bytes/px${over ? '  OVER 2000px' : ''}${blank ? '  LIKELY BLANK' : ''}`)
    for (const e of evidence) console.log(`      asserted ${e}`)
    wrote.push({ file, w, h, mbpp, over, blank })
    if (blank || over) throw new Error(`frame ${file}: fails the frame gate (blank=${blank} over=${over})`)
  }

  async function load() {
    await page.addInitScript(([slot, project, tabsJson]) => {
      localStorage.clear()
      localStorage.setItem('mc-theme', 'dark')
      localStorage.setItem('mc-onboarded', '1')
      localStorage.setItem('mc-active-slot-chat', slot)
      localStorage.setItem('mc-activity-open:' + slot, 'true')
      localStorage.setItem('mc-panel-tabs:' + slot, tabsJson)
      localStorage.setItem('mc-files-rail-open', '1')
      localStorage.setItem('mc-files-rail-w', '360')
      localStorage.setItem('mc-side-panel-width', '820')
      localStorage.setItem('mc-git-panel-opened:' + slot + ':' + project, '1')
      localStorage.setItem('mc-chat-config', JSON.stringify({ pinLastPrompt: false, streamMode: 'immediate' }))
    }, [SLOT, PROJECT, bucket([FILES_TAB], 'files')])
    await page.goto(base + '/?sid=' + encodeURIComponent(SLOT), { waitUntil: 'domcontentloaded' })
    await page.waitForTimeout(2600)
  }

  const panel = () => page.locator('div:has(> .side-panel-strip)').last()

  /** Rows of the tree's shadow root, as [{path, parent, expanded, text}]. */
  const rows = () => page.evaluate(() => {
    const root = document.querySelector('file-tree-container')?.shadowRoot
    if (!root) return null
    return [...root.querySelectorAll('[data-type="item"]:not([data-file-tree-sticky-row])')].map(el => ({
      path: el.getAttribute('data-item-path'),
      parent: el.getAttribute('data-item-parent-path'),
      expanded: el.getAttribute('aria-expanded'),
      // The row's name: `aria-label` is the whole label, where the content
      // section paints it twice (visible + MiddleTruncate's measurement layer).
      text: el.getAttribute('aria-label') ?? '',
    }))
  })
  const waitRows = async (pred, what) => {
    const deadline = Date.now() + 20000
    for (;;) {
      const r = await rows()
      if (r && pred(r)) return r
      if (Date.now() > deadline) throw new Error(`timed out waiting for ${what}; rows=${JSON.stringify(r)}`)
      await page.waitForTimeout(150)
    }
  }
  const dirRow = (path) => page.locator(`file-tree-container [data-type="item"][data-item-path="${path}"]:not([data-file-tree-sticky-row])`).first()
  const expand = async (path) => {
    await dirRow(path).click()
    await waitRows(r => r.some(x => x.path === path && x.expanded === 'true'), `${path} expanded`)
  }
  const collapse = async (path) => {
    await dirRow(path).click()
    await waitRows(r => r.some(x => x.path === path && x.expanded === 'false'), `${path} collapsed`)
  }
  const childrenOf = (r, path) => r.filter(x => x.parent === path)

  async function shot(name, evidence) {
    const file = `${OUT}/${PREFIX}-${name}.png`
    await panel().screenshot({ path: file })
    record(file, evidence)
  }

  // ── Frame 10: five folders expanded ───────────────────────────────────────
  await load()
  await panel().waitFor({ state: 'visible', timeout: 20000 })
  await waitRows(r => r.some(x => x.path === '_bg/'), 'the tree to paint _bg')
  // `vault/` holds only `locked/`, so Pierre's `flattenEmptyDirectories` paints
  // the pair as ONE row whose path is the terminal directory: there is no
  // `vault/` row to expand, and none to call empty.
  for (const p of ['_bg/', 'empty/', 'big/', 'vault/locked/', 'src/']) await expand(p)
  await page.waitForTimeout(600)
  let r = await rows()
  const evidence10 = []
  for (const [path, label] of Object.entries(STATE_LABELS)) {
    const kids = childrenOf(r, path)
    if (EXPECT_STATE_ROWS) {
      if (kids.length !== 1 || !isStateRow(kids[0], label)) {
        throw new Error(`frame 10: expected exactly one state row "${label}" under ${path}, got ${JSON.stringify(kids)}`)
      }
      evidence10.push(`${path} expanded → one state row "${label}" (${kids[0].path}, name ends in the U+200B marker)`)
    } else {
      if (kids.length !== 0) throw new Error(`frame 10 (pre-fix dist): expected NO rows under ${path}, got ${JSON.stringify(kids)}`)
      evidence10.push(`${path} aria-expanded=true → 0 rows beneath it`)
    }
  }
  if (r.some(x => x.path === 'vault/')) {
    throw new Error(`frame 10: vault/ must be flattened into vault/locked/, got its own row: ${JSON.stringify(r.filter(x => x.path === 'vault/'))}`)
  }
  evidence10.push('vault/ has no row of its own: flattened into vault/locked/ (its only entry), so nothing can call it empty')
  const srcKids = childrenOf(r, 'src/')
  if (srcKids.length !== 2 || srcKids.some(k => !/\.py$/.test(k.text))) {
    throw new Error(`frame 10: src/ must show its two files and no state row, got ${JSON.stringify(srcKids)}`)
  }
  evidence10.push(`src/ expanded → ${srcKids.map(k => k.text).join(', ')} and no state row`)
  await shot('10-folders-expanded', evidence10)

  // ── Frame 11: collapse _bg again -- the state row must go with it ─────────
  await collapse('_bg/')
  await page.waitForTimeout(400)
  r = await rows()
  if (childrenOf(r, '_bg/').length !== 0) throw new Error(`frame 11: rows still under a collapsed _bg: ${JSON.stringify(childrenOf(r, '_bg/'))}`)
  const stillEmpty = childrenOf(r, 'empty/')
  if (EXPECT_STATE_ROWS && (stillEmpty.length !== 1 || !isStateRow(stillEmpty[0], STATE_LABELS['empty/']))) {
    throw new Error(`frame 11: empty/ lost its state row: ${JSON.stringify(stillEmpty)}`)
  }
  await shot('11-bg-collapsed', [
    '_bg/ aria-expanded=false → 0 rows beneath it',
    EXPECT_STATE_ROWS ? `empty/ still shows "${STATE_LABELS['empty/']}"` : 'empty/ still shows nothing (pre-fix dist)',
  ])

  // ── Frame 20: the listing request fails ───────────────────────────────────
  // NOT changed by this PR -- captured to settle the "can the fetch fail
  // silently?" question in the issue: every host gates the tree on
  // `useTreeState` and renders its own notice + Refresh in the tree's place.
  treeFails = true
  await load()
  await panel().waitFor({ state: 'visible', timeout: 20000 })
  // One retry with a 1 s backoff (api/queryClient.ts) before the error settles.
  await page.waitForTimeout(4500)
  const failed = await page.evaluate(() => {
    const panelEl = document.querySelector('div > .side-panel-strip')?.parentElement
    const text = (panelEl?.innerText ?? '').replace(/\s+/g, ' ')
    return {
      notice: text.includes("Couldn't load the file tree"),
      refresh: [...(panelEl?.querySelectorAll('button') ?? [])].some(b => b.textContent?.trim() === 'Refresh'),
      skeleton: !!panelEl?.querySelector('[role="status"][aria-label="Loading workspace…"]'),
      tree: !!document.querySelector('file-tree-container'),
      text: text.slice(0, 200),
    }
  })
  console.log('DIAG listing-failed', JSON.stringify(failed))
  if (!failed.notice || !failed.refresh || failed.skeleton || failed.tree) {
    throw new Error(`frame 20: expected the host's "Couldn't load the file tree" notice with Refresh, no skeleton and no tree, got ${JSON.stringify(failed)}`)
  }
  await shot('20-listing-failed', [`host notice "Couldn't load the file tree" with Refresh; no skeleton, no tree mounted (text: ${failed.text})`])

  console.log('\n── SUMMARY ─────────────────────────────')
  for (const w of wrote) console.log(` ok   ${w.w}x${w.h}  ${String(w.mbpp).padStart(4)} mB/px  ${w.file}`)
  await browser.close()
  srv.close()
}

main().catch(err => { console.error(err); process.exit(1) })
