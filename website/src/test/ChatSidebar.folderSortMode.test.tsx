/**
 * Chat sidebar folder order follows `dashboard.folder_sort`.
 *
 * The reporter numbered folders `01.`, `02.`, ..., `98.`, `99.` and the sidebar
 * drew them scrambled, because a placed folder's stored `order` beats its name.
 * The fix is a per-user VIEW preference with three modes -- Custom (the stored
 * positions, today's order and the default), Name (natural order, so 01. < 02. <
 * 10.) and Created (newest first) -- stored server-side so the MCP tree agrees, and
 * chosen from a "Folder order" section of the sidebar's sort-and-filter menu.
 * Choosing a mode never rewrites a stored position.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'
import { render, fireEvent, waitFor, act } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { Provider } from 'react-redux'
import { MemoryRouter } from 'react-router-dom'
import { createTestStore } from './helpers'
import { ThemeProvider } from '../hooks/useTheme'

// Captures the sidebar's drag-end handler so a case can script a folder drop
// without driving dnd-kit's sensors through jsdom (the technique of
// ChatSidebar.dragFreezeOrder.test.tsx). Every DndContext in the sidebar is
// wired to the same handler, so the last one rendered is as good as any. The
// per-row `useSortable` arguments are captured too: whether a folder row is a
// reorder TARGET is decided there (`disabled.droppable`), which no DOM query
// can see.
const dnd = vi.hoisted(() => ({
  handlers: {} as Record<string, ((e: unknown) => void) | undefined>,
  sortables: new Map<string, { disabled?: boolean | { draggable?: boolean; droppable?: boolean } }>(),
}))
vi.mock('@dnd-kit/core', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@dnd-kit/core')>()
  return {
    ...actual,
    DndContext: (props: { children?: unknown; onDragEnd?: (e: unknown) => void }) => {
      dnd.handlers.onDragEnd = props.onDragEnd
      return props.children as never
    },
  }
})
vi.mock('@dnd-kit/sortable', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@dnd-kit/sortable')>()
  return {
    ...actual,
    useSortable: (args: Parameters<typeof actual.useSortable>[0]) => {
      const data = args.data as { type?: string } | undefined
      if (data?.type === 'folder') dnd.sortables.set(String(args.id), { disabled: args.disabled })
      return actual.useSortable(args)
    },
  }
})

// Render framer-motion elements as plain DOM (jsdom can't run projection).
vi.mock('framer-motion', async () => {
  const React = await import('react')
  const FRAMER_PROPS = new Set([
    'layout', 'layoutId', 'layoutScroll', 'initial', 'animate', 'exit',
    'transition', 'variants', 'whileHover', 'whileTap', 'whileInView',
    'drag', 'dragConstraints', 'dragElastic', 'onAnimationComplete',
  ])
  const make = (tag: string) =>
    React.forwardRef((props: Record<string, unknown>, ref: React.Ref<unknown>) => {
      const clean: Record<string, unknown> = {}
      for (const k of Object.keys(props)) {
        if (k === 'children') continue
        if (k === 'layoutId') { clean['data-layout-id'] = props[k]; continue }
        if (FRAMER_PROPS.has(k)) continue
        clean[k] = props[k]
      }
      return React.createElement(tag, { ...clean, ref }, props.children as React.ReactNode)
    })
  const motion = new Proxy({}, { get: (_t, tag: string) => make(tag) })
  return {
    motion,
    AnimatePresence: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
    LayoutGroup: ({ children }: { children?: React.ReactNode }) => React.createElement(React.Fragment, null, children),
  }
})

vi.mock('../components/ProjectPicker', () => ({ default: () => null }))
vi.mock('../pages/chat/ChatSettings', () => ({
  loadChatConfig: () => ({ tagColumnsEnabled: false, confirmCloseSession: false }),
  saveChatConfig: vi.fn(),
}))

/** The gateway's config as the test's fake server holds it. `patchConfig` writes
 *  into it and `kirocrewConfig` reads it back, so the settle-time refetch the
 *  optimistic overlay triggers after a save sees the value the save landed --
 *  exactly the round trip the real endpoint pair performs. */
const serverConfig: { dashboard: Record<string, unknown> } = { dashboard: {} }
const patchConfig = vi.fn(async (path: string, value: unknown) => {
  if (path === 'dashboard.folder_sort') serverConfig.dashboard.folder_sort = value
  return { ok: true }
})
const kirocrewConfig = vi.fn(async () => ({ dashboard: { ...serverConfig.dashboard } }))
vi.mock('../api/client', () => ({
  SEARCH_MIN_CHARS: 2,
  api: new Proxy({} as Record<string, unknown>, {
    get: (_t, name: string) => {
      if (name === 'patchConfig') return patchConfig
      if (name === 'kirocrewConfig') return kirocrewConfig
      return vi.fn().mockResolvedValue([])
    },
  }),
}))

Object.defineProperty(window, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation((q: string) => ({
    matches: false, media: q, onchange: null,
    addListener: vi.fn(), removeListener: vi.fn(),
    addEventListener: vi.fn(), removeEventListener: vi.fn(), dispatchEvent: vi.fn(),
  })),
})

import ChatSidebar from '../pages/ChatSidebar'
import type { RootState } from '../store'
import type { ChatFolder, ChatSlot } from '../types'
import { bySidebarOrder } from '../utils/folderTree'

/** The reporter's scheme with the stored positions a few drags left behind:
 *  the stored order reads 10, 99, 98, 02, 03, 01 -- the sidebar he screenshotted. */
const NUMBERED: ChatFolder[] = [
  { id: 'f10', name: '10. Zulu', order: 0, collapsed: true, created_at: 1_000 },
  { id: 'f99', name: '99. Omega', order: 1, collapsed: true, created_at: 6_000 },
  { id: 'f98', name: '98. Tango', order: 2, collapsed: true, created_at: 5_000 },
  { id: 'f02', name: '02. Mike', order: 3, collapsed: true, created_at: 2_000 },
  { id: 'f03', name: '03. Kilo', order: 4, collapsed: true, created_at: 3_000 },
  { id: 'f01', name: '01. Alpha', order: 5, collapsed: true, created_at: 4_000 },
]

function renderSidebar(folders: ChatFolder[], folderSort: unknown, opts: { configReadFails?: Error } = {}) {
  const slots: ChatSlot[] = []
  const store = createTestStore({
    dashboard: {
      status: {}, connected: true, slots, approvalMode: 'normal',
      channelTrusted: false, refreshTrigger: 0, unreadSlots: [], updateProgress: null,
      slotsLoaded: true,
      subagentRunning: {}, subagentDetails: {}, subagentText: {},
      sessionDefaultColor: null, sessionColorsMode: 'tint', sessionColorsPalette: 'horizon', sessionColorsIntensity: 'clear',
    } as unknown as RootState['dashboard'],
    chat: { activeSlot: null, slotStatusDetail: {}, subagents: {}, slotActivity: {}, workflowRuns: {} } as unknown as RootState['chat'],
  })
  // staleTime keeps both seeded caches authoritative: the blanket api mock
  // resolves every read to [], so an on-mount refetch would wipe the folders and
  // the config out from under the rows we are asserting on.
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity, refetchOnMount: false }, mutations: { retry: false } } })
  qc.setQueryData(['chat-folders'], folders)
  serverConfig.dashboard = folderSort === undefined ? {} : { folder_sort: folderSort }
  if (opts.configReadFails) {
    // No seeded config: the mount fetch is the read, and it fails -- the query
    // settles in its error state (retry is off) with no data to fall back on.
    kirocrewConfig.mockRejectedValueOnce(opts.configReadFails)
  } else {
    qc.setQueryData(['kirocrewConfig'], { dashboard: { ...serverConfig.dashboard } })
  }
  const utils = render(
    <QueryClientProvider client={qc}>
      <Provider store={store}>
        <ThemeProvider>
          <MemoryRouter>
            <ChatSidebar
              slots={slots} activeSlot={null} unreadSlots={[]}
              history={[]} historyHasMore={false} defaultAgent="" installedAgents={[]}
            />
          </MemoryRouter>
        </ThemeProvider>
      </Provider>
    </QueryClientProvider>,
  )
  return { ...utils, qc }
}

/** Folder ids in the order the tree draws their header rows. */
function drawnFolderIds(container: HTMLElement): string[] {
  return [...container.querySelectorAll('[data-folder-row]')].map(el => el.getAttribute('data-folder-row') ?? '')
}

beforeEach(() => { localStorage.clear(); patchConfig.mockClear(); kirocrewConfig.mockClear(); dnd.sortables.clear() })
afterEach(() => vi.clearAllMocks())

describe('chat sidebar — folder order follows dashboard.folder_sort', () => {
  it('draws the stored order in custom mode -- byte-identical to bySidebarOrder', () => {
    const { container } = renderSidebar(NUMBERED, 'custom')
    expect(drawnFolderIds(container)).toEqual([...NUMBERED].sort(bySidebarOrder).map(f => f.id))
    expect(drawnFolderIds(container)).toEqual(['f10', 'f99', 'f98', 'f02', 'f03', 'f01'])
  })

  it('treats an absent or unknown stored value as custom, so an upgrade changes nothing', () => {
    expect(drawnFolderIds(renderSidebar(NUMBERED, undefined).container)).toEqual(['f10', 'f99', 'f98', 'f02', 'f03', 'f01'])
    expect(drawnFolderIds(renderSidebar(NUMBERED, 'alphabetical').container)).toEqual(['f10', 'f99', 'f98', 'f02', 'f03', 'f01'])
  })

  it('draws the numbered folders in natural order in name mode', () => {
    const { container } = renderSidebar(NUMBERED, 'name')
    expect(drawnFolderIds(container)).toEqual(['f01', 'f02', 'f03', 'f10', 'f98', 'f99'])
  })

  it('draws newest first in created mode', () => {
    const { container } = renderSidebar(NUMBERED, 'created')
    expect(drawnFolderIds(container)).toEqual(['f99', 'f98', 'f01', 'f03', 'f02', 'f10'])
  })

  it('applies the mode at every depth, not only to root folders', () => {
    const nested: ChatFolder[] = [
      { id: 'root', name: 'Projects', order: 0, collapsed: false },
      { id: 'c10', name: '10. late', order: 0, parent_id: 'root', collapsed: true },
      { id: 'c2', name: '2. early', order: 1, parent_id: 'root', collapsed: true },
    ]
    expect(drawnFolderIds(renderSidebar(nested, 'custom').container)).toEqual(['root', 'c10', 'c2'])
    expect(drawnFolderIds(renderSidebar(nested, 'name').container)).toEqual(['root', 'c2', 'c10'])
  })

  it('offers the three modes under "Folder order" in the sort-and-filter menu and marks the active one', async () => {
    const { getByLabelText, findByTestId, getByTestId } = renderSidebar(NUMBERED, 'custom')
    fireEvent.keyDown(getByLabelText('Sort and filter sessions'), { key: 'Enter' })
    const custom = await findByTestId('folder-order-custom')
    expect(custom.textContent).toContain('Custom')
    expect(getByTestId('folder-order-name').textContent).toContain('Name')
    expect(getByTestId('folder-order-created').textContent).toContain('Created (Newest)')
    // The check mark sits on the active row only.
    expect(custom.querySelector('svg.text-accent')).toBeTruthy()
    expect(getByTestId('folder-order-name').querySelector('svg.text-accent')).toBeNull()
  })

  it('picking Name PATCHes dashboard.folder_sort once, re-sorts the tree at once, and rewrites no folder', async () => {
    const { getByLabelText, findByTestId, container, qc } = renderSidebar(NUMBERED, 'custom')
    fireEvent.keyDown(getByLabelText('Sort and filter sessions'), { key: 'Enter' })
    fireEvent.click(await findByTestId('folder-order-name'))
    await waitFor(() => expect(patchConfig).toHaveBeenCalledWith('dashboard.folder_sort', 'name'))
    expect(patchConfig).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(drawnFolderIds(container)).toEqual(['f01', 'f02', 'f03', 'f10', 'f98', 'f99']))
    // A VIEW change: the folder rows themselves are untouched, so switching back to
    // Custom restores exactly the arrangement the person had.
    expect(qc.getQueryData<ChatFolder[]>(['chat-folders'])).toEqual(NUMBERED)
    expect(patchConfig.mock.calls.every(([path]) => path === 'dashboard.folder_sort')).toBe(true)
  })

  it('picking the active mode again writes nothing', async () => {
    const { getByLabelText, findByTestId } = renderSidebar(NUMBERED, 'name')
    fireEvent.keyDown(getByLabelText('Sort and filter sessions'), { key: 'Enter' })
    fireEvent.click(await findByTestId('folder-order-name'))
    expect(patchConfig).not.toHaveBeenCalled()
  })

  it('a refused save is said on the folder-action notice and the tree returns to the stored order', async () => {
    patchConfig.mockRejectedValueOnce(new Error('governance refused this write'))
    const { getByLabelText, findByTestId, container } = renderSidebar(NUMBERED, 'custom')
    fireEvent.keyDown(getByLabelText('Sort and filter sessions'), { key: 'Enter' })
    fireEvent.click(await findByTestId('folder-order-name'))
    // The overlay rolls the display back on its own; the notice is what tells the
    // person why the tree snapped back.
    const notice = await findByTestId('folder-action-error')
    expect(notice.textContent).toContain('governance refused this write')
    await waitFor(() => expect(drawnFolderIds(container)).toEqual(['f10', 'f99', 'f98', 'f02', 'f03', 'f01']))
    // The refused write never reached the server's copy.
    expect(serverConfig.dashboard.folder_sort).toBe('custom')
  })

  it('outside Custom a folder row is no reorder target, and a scripted sibling drop writes nothing', async () => {
    const first = renderSidebar(NUMBERED, 'name')
    // The affordance is withdrawn at the row: with the droppable side off, no
    // pointer drag can resolve a sibling as `over`, so no slot ever opens.
    expect(dnd.sortables.get('f10')?.disabled).toEqual({ droppable: true })
    expect(dnd.sortables.get('f99')?.disabled).toEqual({ droppable: true })
    // The belt for the keyboard/scripted path: the handler declines the write
    // silently -- an expected restriction, not a failed update.
    expect(typeof dnd.handlers.onDragEnd).toBe('function')
    const drop = {
      active: { id: 'f10', data: { current: { type: 'folder' } } },
      over: { id: 'f99', data: { current: { type: 'folder' } } },
    }
    act(() => { dnd.handlers.onDragEnd!(drop) })
    expect(first.queryByTestId('folder-action-error')).toBeNull()
    expect(first.qc.getQueryData<ChatFolder[]>(['chat-folders'])).toEqual(NUMBERED)
    first.unmount()

    // In Custom the rows are ordinary sortables again.
    dnd.sortables.clear()
    renderSidebar(NUMBERED, 'custom')
    expect(dnd.sortables.get('f10')?.disabled).toBeUndefined()
  })

  it('while the folder order cannot be read, the tree draws the stored order, says so, and withdraws reordering', async () => {
    const { container, findByTestId, queryByTestId } = renderSidebar(NUMBERED, 'name', { configReadFails: new Error('gateway restarting') })
    // The stored order is the fallback -- every earlier build drew it -- and an
    // unknown mode is never treated as a writable Custom.
    expect(drawnFolderIds(container)).toEqual(['f10', 'f99', 'f98', 'f02', 'f03', 'f01'])
    expect(dnd.sortables.get('f10')?.disabled).toEqual({ droppable: true })
    const notice = await findByTestId('folder-order-unavailable')
    expect(notice.textContent).toContain('Folder order could not be read')
    expect(notice.textContent).toContain('gateway restarting')
    expect(dnd.sortables.get('f10')?.disabled).toEqual({ droppable: true })
    expect(queryByTestId('folder-action-error')).toBeNull()
  })
})
