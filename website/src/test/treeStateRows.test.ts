/**
 * `planTreeStateRows` decides which folders of a listing get the tree's state
 * row and what it says; `PIERRE_TREE_STATE_ROW_CSS` selects those rows in
 * Pierre's shadow root by the marker every synthetic segment ends in. Both are
 * pure, so the edge cases the component tests would only reach through a
 * payload are pinned here.
 */
import { describe, it, expect } from 'vitest'
import { planTreeStateRows, stateRowSegment, STATE_ROW_MARKER } from '../pierre/treeStateRows'
import { PIERRE_TREE_STATE_ROW_CSS } from '../pierre/config'

const LABELS = {
  empty: 'Empty folder',
  'hidden-only': 'No visible files',
  unreadable: 'Folder not readable',
  truncated: 'Files not shown',
} as const
const M = STATE_ROW_MARKER

describe('planTreeStateRows', () => {
  it('names one row per childless folder, with the kind the payload proves', () => {
    const plan = planTreeStateRows(
      {
        paths: ['src/a.ts'],
        directories: ['src', 'empty', 'onlyhidden', 'locked', 'cut'],
        hiddenOnlyDirectories: ['onlyhidden'],
        unreadableDirectories: ['locked'],
        truncatedDirectories: ['cut'],
      },
      LABELS,
    )
    expect(plan.paths).toEqual([
      `empty/Empty folder${M}`,
      `onlyhidden/No visible files${M}`,
      `locked/Folder not readable${M}`,
      `cut/Files not shown${M}`,
    ])
    expect(plan.kinds.get(`empty/Empty folder${M}`)).toBe('empty')
    expect(plan.kinds.get(`onlyhidden/No visible files${M}`)).toBe('hidden-only')
    expect(plan.kinds.get(`locked/Folder not readable${M}`)).toBe('unreadable')
    expect(plan.kinds.get(`cut/Files not shown${M}`)).toBe('truncated')
  })

  it('shows an unreadable folder as a folder, so its parent is not called empty', () => {
    // The server lists the folder it could not read as a directory row of its
    // own; the row beneath it carries the qualifier, and `vault` -- whose only
    // entry it is -- has a child and gets no row at all.
    const plan = planTreeStateRows(
      { paths: [], directories: ['vault', 'vault/locked'], unreadableDirectories: ['vault/locked'] },
      LABELS,
    )
    expect(plan.paths).toEqual([`vault/locked/Folder not readable${M}`])
    expect(plan.kinds.get(`vault/locked/Folder not readable${M}`)).toBe('unreadable')
  })

  it('treats a folder with a subfolder as populated, down to the childless leaf', () => {
    const plan = planTreeStateRows({ paths: [], directories: ['a/b/c'] }, LABELS)
    expect(plan.paths).toEqual([`a/b/c/Empty folder${M}`])
  })

  it('reads implicit ancestors off file paths and explicit rows alike', () => {
    // `pkg` is named only as an ancestor of a file; `docs/` arrives with the
    // server's trailing slash. Neither is childless.
    const plan = planTreeStateRows({ paths: ['pkg/lib/x.ts'], directories: ['docs/', 'docs/api'] }, LABELS)
    expect(plan.paths).toEqual([`docs/api/Empty folder${M}`])
  })

  it('puts nothing under a listing with no folders', () => {
    expect(planTreeStateRows({ paths: ['a.ts', 'b.ts'] }, LABELS).paths).toEqual([])
  })

  it('never lets a label mint a subfolder', () => {
    // A translation written with a slash would otherwise become two segments
    // and the row would land one level too deep, under a folder that does not
    // exist.
    expect(stateRowSegment('no files / hidden')).toBe(`no files \u2215 hidden${M}`)
    const plan = planTreeStateRows(
      { paths: [], directories: ['d'] },
      { ...LABELS, empty: 'vide / rien' },
    )
    expect(plan.paths).toEqual([`d/vide \u2215 rien${M}`])
  })

  it('ends every segment in the marker the stylesheet selects, and paints nothing for it', () => {
    // The marker is what tells a state row from a real file that happens to be
    // named like a label; it must be invisible and must not be whitespace the
    // widget could trim away.
    expect(stateRowSegment('Empty folder')).toBe(`Empty folder${M}`)
    expect(M).toBe('\u200b')
    expect(M.trim()).toBe(M)
  })
})

describe('PIERRE_TREE_STATE_ROW_CSS', () => {
  it('selects the marker as a path suffix, never a label, and keeps the pointer off the row', () => {
    expect(PIERRE_TREE_STATE_ROW_CSS).toContain(`[data-type="item"][data-item-path$="${M}"]`)
    expect(PIERRE_TREE_STATE_ROW_CSS).not.toContain('Empty folder')
    expect(PIERRE_TREE_STATE_ROW_CSS).toContain('pointer-events:none')
    expect(PIERRE_TREE_STATE_ROW_CSS).toContain('[data-item-section="icon"]')
    expect(PIERRE_TREE_STATE_ROW_CSS).toContain('[data-item-section="action"]')
  })
})
