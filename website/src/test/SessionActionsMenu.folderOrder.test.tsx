/** The session menu's move-to submenu lists chat folders in the sidebar's folder
 *  order (`dashboard.folder_sort`). When the settings read behind that order
 *  fails, the submenu is drawn in the stored order -- a different list than the
 *  one the person chose -- so the menu says so, in the Radix menu form: a passive
 *  alert with the server's own words, and the agent hand-off as a sibling menu
 *  item the roving focus can reach, described by that alert.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders, createTestStore } from './helpers'
import { sseSlots } from '../store/dashboardSlice'
import {
  consumeChatHandoff,
  installSoftNavigate,
  __resetErrorJournalForTests,
  __resetNavSeamForTests,
} from '../utils/errorReport'
import type { ChatFolder, ChatSlot } from '../types'
import SessionActionsMenu from '../components/SessionActionsMenu'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuTrigger,
} from '../components/ui/context-menu'

const mocks = vi.hoisted(() => ({
  chatFolders: vi.fn(),
  kirocrewConfig: vi.fn(),
}))

vi.mock('../api/client', () => ({
  api: new Proxy(mocks as Record<string, unknown>, {
    get: (target, property: string) => (
      property in target ? target[property] : vi.fn().mockResolvedValue([])
    ),
  }),
}))

// The submenu itself is a Radix Sub, flaky under jsdom; its presence is what
// this file gates on, so a stub that renders a marker is enough.
vi.mock('../components/FolderMoveSubmenu', () => ({ default: () => <div data-testid="move-submenu" /> }))
vi.mock('../components/SendToInstanceSubmenu', () => ({ default: () => null }))
vi.mock('../components/SessionColorSwatches', () => ({ default: () => null }))
vi.mock('../components/LinkedSurfacesSection', () => ({ default: () => null }))
vi.mock('../components/ExportSessionItem', () => ({ default: () => null }))
vi.mock('../components/ImportSessionItem', () => ({ default: () => null }))
vi.mock('../hooks/useSessionActions', () => ({
  useSessionActions: () => ({
    toggleRead: vi.fn(),
    togglePin: vi.fn(),
    toggleMode: vi.fn(),
    copyLink: vi.fn(),
    move: vi.fn(),
    reload: vi.fn(),
    close: vi.fn(),
  }),
}))
vi.mock('../hooks/useChatPopouts', () => ({
  useChatPopouts: () => ({
    isPoppedOut: () => false,
    isSelfPopout: () => false,
    open: vi.fn(),
    focus: vi.fn(),
    bringBack: vi.fn(),
    returnSelfToMain: vi.fn(),
  }),
}))
vi.mock('../hooks/useTagPopover', () => ({
  useTagPopover: () => ({ open: vi.fn() }),
}))

const FOLDERS: ChatFolder[] = [{ id: 'work', name: 'Work', order: 0 }]

function mount() {
  const store = createTestStore()
  store.dispatch(sseSlots([{
    key: 'context-slot',
    messages: 1,
    running: false,
    memory_mode: 'persistent',
  } as ChatSlot]))
  const view = renderWithProviders(
    <ContextMenu>
      <ContextMenuTrigger asChild>
        <button type="button" data-testid="context-trigger">Actions</button>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <SessionActionsMenu variant="context" slotKey="context-slot" />
      </ContextMenuContent>
    </ContextMenu>,
    { store },
  )
  fireEvent.contextMenu(screen.getByTestId('context-trigger'))
  return view
}

beforeEach(() => {
  vi.clearAllMocks()
  mocks.chatFolders.mockResolvedValue(FOLDERS)
  mocks.kirocrewConfig.mockResolvedValue({ dashboard: { folder_sort: 'name' } })
  __resetErrorJournalForTests()
  __resetNavSeamForTests()
  sessionStorage.clear()
  installSoftNavigate(() => {})
})

afterEach(() => {
  __resetNavSeamForTests()
  vi.restoreAllMocks()
})

describe('SessionActionsMenu folder-order read failure', () => {
  it('says nothing while the order reads fine', async () => {
    mount()
    await screen.findByTestId('move-submenu')
    expect(screen.queryByTestId('session-menu-folder-order-unavailable')).toBeNull()
    expect(screen.queryByRole('menuitem', { name: /^ask the agent$/i })).toBeNull()
  })

  it('says the order could not be read, and hands the server string to the agent from a sibling item', async () => {
    const user = userEvent.setup()
    mocks.kirocrewConfig.mockRejectedValue(new Error('gateway restarting'))
    mount()

    // The passive alert carries the server's own words under the localized lead,
    // beside the submenu whose order it explains.
    const notice = await screen.findByTestId('session-menu-folder-order-unavailable')
    expect(notice).toHaveAttribute('role', 'alert')
    expect(notice).toHaveTextContent('Folder order could not be read')
    expect(notice).toHaveTextContent('gateway restarting')
    expect(screen.getByTestId('move-submenu')).toBeInTheDocument()

    // The hand-off is a real menu item (a button nested in an item is skipped by
    // the menu's roving focus), described by the alert it acts on.
    const handoff = screen.getByRole('menuitem', { name: /^ask the agent$/i })
    expect(notice.id).not.toBe('')
    expect(handoff).toHaveAttribute('aria-describedby', notice.id)
    handoff.focus()
    await user.keyboard('{Enter}')
    expect(consumeChatHandoff()).toContain('gateway restarting')
  })

  it('says nothing without folders: there is no submenu whose order it would explain', async () => {
    mocks.chatFolders.mockResolvedValue([])
    mocks.kirocrewConfig.mockRejectedValue(new Error('gateway restarting'))
    mount()
    await screen.findByRole('menuitem', { name: /tags/i })
    expect(screen.queryByTestId('move-submenu')).toBeNull()
    expect(screen.queryByTestId('session-menu-folder-order-unavailable')).toBeNull()
  })
})
