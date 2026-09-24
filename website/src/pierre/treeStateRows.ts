/**
 * The one row the workspace tree puts under an expanded folder that has
 * nothing beneath it.
 *
 * The listing (`GET /api/project/tree`) arrives whole: there is no per-folder
 * request, so whether a folder has children is decided by the payload the
 * moment it lands, never by a fetch in flight. What the payload can leave
 * childless is a folder that is
 *
 *  - `empty`: listed as a directory with no file and no subfolder in it;
 *  - `hidden-only`: not empty on disk, but every entry in it is one the
 *    listing hides (a dot-directory, a tooling cache) -- the server names these
 *    in `hiddenOnlyDirectories`, and without that signal `_bg/` holding only
 *    `.kiro/` would be called empty;
 *  - `unreadable`: the server could not read it at all (permission denied), so
 *    nothing beneath it is known -- named in `unreadableDirectories`; without
 *    that row the folder would vanish from the tree and its parent, holding
 *    nothing else, would be called empty;
 *  - `truncated`: it has files, but the workspace file cap left it none --
 *    the server names these in `truncatedDirectories`.
 *
 * `@pierre/trees` renders rows for paths and has no slot for a status line, so
 * each state row is fed to the model as ONE synthetic child path whose
 * basename is the label followed by `STATE_ROW_MARKER`. That buys the widget's
 * own indentation, virtualization and keyboard reachability for free; the
 * wrapper keeps the row inert (no open, no selection, no context menu) and
 * `PIERRE_TREE_STATE_ROW_CSS` styles it as a status line. Known cost, accepted:
 * the label is a real path to the library, so the tree's search matches it like
 * any other name.
 */

export type TreeStateRowKind = 'empty' | 'hidden-only' | 'unreadable' | 'truncated'

export type TreeStateRowLabels = Readonly<Record<TreeStateRowKind, string>>

export interface TreeStateRowPlan {
  /** Synthetic child paths, one per childless folder, in folder order. */
  paths: string[]
  /** Which state each synthetic path stands for, keyed by that path. */
  kinds: Map<string, TreeStateRowKind>
}

/** The payload fields the plan reads; a subset of `api.projectTree`'s result. */
export interface TreeStateRowPayload {
  paths: readonly string[]
  directories?: readonly string[]
  truncatedDirectories?: readonly string[]
  hiddenOnlyDirectories?: readonly string[]
  unreadableDirectories?: readonly string[]
}

const parentOf = (path: string): string => {
  const cut = path.lastIndexOf('/')
  return cut === -1 ? '' : path.slice(0, cut)
}

/**
 * Zero-width space: paints nothing, screen readers skip it, and no real name
 * ends in one. Trailing, so the selector is one `data-item-path$=` suffix.
 */
export const STATE_ROW_MARKER = '\u200b'

/**
 * A label becomes a path SEGMENT, so it must not contain the separator: a
 * translation written as "no files / hidden" would otherwise mint a subfolder.
 * The division slash (U+2215) reads the same and is an ordinary name character.
 *
 * The segment then ends in `STATE_ROW_MARKER`, and that marker -- not the label
 * -- is what `PIERRE_TREE_STATE_ROW_CSS` selects: a real file that happens to be
 * named like a label ("Empty folder", "Dossier vide") must keep its icon and
 * its pointer, and the stylesheet is fixed at model construction while the
 * labels follow the active language.
 */
export function stateRowSegment(label: string): string {
  return label.replace(/\//g, '\u2215') + STATE_ROW_MARKER
}

export function planTreeStateRows(tree: TreeStateRowPayload, labels: TreeStateRowLabels): TreeStateRowPlan {
  // Every directory the payload knows: the explicit skeleton (which the server
  // sends even for folders whose files were cut) plus every ancestor of a
  // listed file or of an explicit directory.
  const directories = new Set<string>()
  const addWithAncestors = (dir: string) => {
    for (let d = dir; d; d = parentOf(d)) {
      if (directories.has(d)) break
      directories.add(d)
    }
  }
  for (const d of tree.directories ?? []) addWithAncestors(d.replace(/\/$/, ''))
  for (const p of tree.paths) addWithAncestors(parentOf(p))

  const withChild = new Set<string>()
  for (const p of tree.paths) withChild.add(parentOf(p))
  for (const d of directories) withChild.add(parentOf(d))

  const truncated = new Set(tree.truncatedDirectories ?? [])
  const unreadable = new Set(tree.unreadableDirectories ?? [])
  const hiddenOnly = new Set(tree.hiddenOnlyDirectories ?? [])
  const segments = {
    empty: stateRowSegment(labels.empty),
    'hidden-only': stateRowSegment(labels['hidden-only']),
    unreadable: stateRowSegment(labels.unreadable),
    truncated: stateRowSegment(labels.truncated),
  } as const

  const plan: TreeStateRowPlan = { paths: [], kinds: new Map() }
  for (const dir of directories) {
    if (withChild.has(dir)) continue
    // Truncation is checked first: a folder that lost its files to the cap is
    // not empty, and the server never lists a folder with files as hidden-only.
    // An unreadable folder was never read, so it is in neither of the other
    // two lists; the order below only fixes what a malformed payload shows.
    const kind: TreeStateRowKind = truncated.has(dir)
      ? 'truncated'
      : unreadable.has(dir)
        ? 'unreadable'
        : hiddenOnly.has(dir)
          ? 'hidden-only'
          : 'empty'
    const path = `${dir}/${segments[kind]}`
    plan.paths.push(path)
    plan.kinds.set(path, kind)
  }
  return plan
}
