import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import McpToolsPanel, { SESSION_DOT_CLASS } from '../pages/chat/McpToolsPanel'

const servers = [{ name: 'slack-mcp', enabled: true }]
const toolsByServer = {
  'slack-mcp': { tools: ['post_message', 'delete_message', 'legacy'], disabledTools: ['legacy'] },
}

describe('McpToolsPanel', () => {
  it('renders the Tool Search mode line (deferred)', () => {
    render(
      <McpToolsPanel servers={servers} toolsByServer={toolsByServer} loaded={new Set()} toolSearchOn={true} loading={false} />,
    )
    expect(screen.getByText('Tool Search · Deferred')).toBeInTheDocument()
  })

  it('shows a per-server loaded/total count and marks each tool loaded / deferred / disabled', () => {
    const loaded = new Set(['slack-mcp::post_message'])
    render(
      <McpToolsPanel servers={servers} toolsByServer={toolsByServer} loaded={loaded} toolSearchOn={true} loading={false} />,
    )
    // 1 of 2 loadable loaded (legacy is disabled → excluded from the denominator;
    // delete_message deferred; post_message loaded)
    expect(screen.getByText('1/2')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /slack-mcp/ }))
    expect(screen.getByText('post_message')).toBeInTheDocument()
    expect(screen.getByTitle('Loaded this session')).toBeInTheDocument()
    expect(screen.getByTitle('Deferred')).toBeInTheDocument()
    expect(screen.getByTitle('Disabled')).toBeInTheDocument()
  })

  // #10320: with no session report this panel had nothing that observed a
  // handshake, yet it painted the configured `enabled` flag -- a read of
  // mcp.json -- as `bg-ok`. That arm now wears the `no_report` mark. The
  // disabled arm keeps its filled muted dot.
  it('marks a configured server "no report" when no session report has arrived', () => {
    render(
      <McpToolsPanel
        servers={[{ name: 'slack-mcp', enabled: true }, { name: 'off-mcp', enabled: false }]}
        toolsByServer={toolsByServer}
        loaded={new Set()}
        toolSearchOn={true}
        loading={false}
      />,
    )
    const dot = (name: string) =>
      screen.getByRole('button', { name: new RegExp(name) }).querySelector('span.rounded-full')!
    expect(dot('slack-mcp').className).not.toContain('bg-ok')
    for (const cls of SESSION_DOT_CLASS.no_report.split(' ')) {
      expect(dot('slack-mcp').className).toContain(cls)
    }
    expect(dot('off-mcp').className).toContain('bg-muted')
    // Named even with no report in hand: the legend that decodes this mark is
    // itself gated on having a report, so the tooltip is the only thing that
    // says what the ring means in the state it appears in.
    expect(dot('slack-mcp')).toHaveAttribute('title', 'No report from this session yet')
  })

  it('marks every non-disabled tool active when tool search is off', () => {
    render(
      <McpToolsPanel servers={servers} toolsByServer={toolsByServer} loaded={new Set()} toolSearchOn={false} loading={false} />,
    )
    expect(screen.getByText('Tool Search · Fully loaded')).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /slack-mcp/ }))
    expect(screen.getAllByTitle('Loaded this session').length).toBe(2)
    expect(screen.getByTitle('Disabled')).toBeInTheDocument()
  })
})
