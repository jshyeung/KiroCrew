import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render as rtlRender, screen, fireEvent, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactElement } from 'react'
import McpInfoButton from '../pages/chat/McpInfoButton'
import { SESSION_DOT_CLASS } from '../pages/chat/McpToolsPanel'
import { api } from '../api/client'

vi.mock('../api/client', () => ({
  api: {
    mcpActive: vi.fn().mockResolvedValue([
      { name: 'builder-mcp', enabled: true },
      { name: 'slack-mcp', enabled: false },
    ]),
    kirocrewConfig: vi.fn().mockResolvedValue({ agent: { tool_search: true } }),
  },
}))

// The popover's two reads are react-query queries (per website/AGENTS.md), so
// the component needs a provider; a fresh client per render keeps one test's
// cached answer out of the next.
function render(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return rtlRender(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>)
}

describe('McpInfoButton', () => {
  beforeEach(() => { vi.clearAllMocks() })

  it('renders info button', () => {
    render(<McpInfoButton />)
    expect(screen.getByTitle('Session MCP servers')).toBeInTheDocument()
  })

  it('shows server list on click', async () => {
    render(<McpInfoButton />)
    fireEvent.click(screen.getByTitle('Session MCP servers'))
    await waitFor(() => {
      expect(screen.getByText('builder-mcp')).toBeInTheDocument()
      expect(screen.getByText('slack-mcp')).toBeInTheDocument()
    })
  })

  it('shows disabled label for disabled servers', async () => {
    render(<McpInfoButton />)
    fireEvent.click(screen.getByTitle('Session MCP servers'))
    await waitFor(() => {
      expect(screen.getByText('disabled')).toBeInTheDocument()
    })
  })

  // #10320: `enabled` comes from GET /api/mcp/active, which reads the spec's
  // `disabled` flag and carries no session handshake, so the dot must not wear
  // the `ok` status hue. Both arms are pinned through the seam the component
  // branches on: a configured row gets the `no_report` mark, a disabled row
  // keeps the filled muted dot.
  // Both surfaces now render the one exported constant, so there is no copied
  // string left to drift; what this pins is that the popover uses the `no_report`
  // vocabulary at all, and that the mark is NAMED -- at 6px an unlabelled dashed
  // ring reads as a spinner, and the legend that would decode it lives on the
  // sibling surface.
  const LAYOUT = ['w-1.5', 'h-1.5', 'rounded-full', 'shrink-0']

  it('marks a configured server "no report", not the ok status hue', async () => {
    render(<McpInfoButton />)
    fireEvent.click(screen.getByTitle('Session MCP servers'))
    await waitFor(() => expect(screen.getByText('builder-mcp')).toBeInTheDocument())
    const mark = (name: string) =>
      screen.getByText(name).parentElement!.querySelector('span.rounded-full')!

    expect(mark('builder-mcp').className).not.toContain('bg-ok')
    expect([...mark('builder-mcp').className.split(/\s+/).filter(Boolean)].sort()).toEqual(
      [...LAYOUT, ...SESSION_DOT_CLASS.no_report.split(/\s+/)].sort(),
    )
    expect(mark('builder-mcp')).toHaveAttribute('title', 'No report from this session yet')
    expect(mark('slack-mcp').className.split(/\s+/).filter(Boolean)).toEqual([...LAYOUT, 'bg-muted'])
    expect(mark('slack-mcp')).not.toHaveAttribute('title')
  })

  it('closes on outside click', async () => {
    render(<McpInfoButton />)
    fireEvent.click(screen.getByTitle('Session MCP servers'))
    await waitFor(() => expect(screen.getByText('builder-mcp')).toBeInTheDocument())
    fireEvent.pointerDown(document.body)
    expect(screen.queryByText('builder-mcp')).not.toBeInTheDocument()
  })

  it('shows the "Deferred" Tool Search status when tool_search is on', async () => {
    vi.mocked(api.kirocrewConfig).mockResolvedValue({ agent: { tool_search: true } })
    render(<McpInfoButton />)
    fireEvent.click(screen.getByTitle('Session MCP servers'))
    await waitFor(() => expect(screen.getByText('Tool Search · Deferred')).toBeInTheDocument())
    expect(screen.queryByText('Tool Search · Fully loaded')).not.toBeInTheDocument()
  })

  it('shows the "Fully loaded" Tool Search status when tool_search is off', async () => {
    vi.mocked(api.kirocrewConfig).mockResolvedValue({ agent: { tool_search: false } })
    render(<McpInfoButton />)
    fireEvent.click(screen.getByTitle('Session MCP servers'))
    await waitFor(() => expect(screen.getByText('Tool Search · Fully loaded')).toBeInTheDocument())
    expect(screen.queryByText('Tool Search · Deferred')).not.toBeInTheDocument()
  })
})
