import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import { i18nT } from '../i18n/t'
import { errMessage } from '../utils/thunkError'
import { readFolderSortMode, type FolderSortMode } from '../utils/folderTree'

export interface FolderSortModeRead {
  /**
   * The mode to draw with: the person's once the config is read, `custom`
   * (the stored order every earlier build drew) while it loads, after it
   * failed, or from an older gateway without the field.
   */
  readonly mode: FolderSortMode
  /**
   * Why `mode` may not be the one the person chose: the shared config read
   * failed. The raw server string (the generic fallback when the failure
   * carried none), so an `ErrorNotice` given it as `message` recovers the
   * structured report from the error journal by message match. `null` while
   * loading and after success. Every host renders it -- a picker silently
   * drawn in the stored order behind a mode the person did choose is the dead
   * end the notice removes.
   */
  readonly error: string | null
}

/**
 * The person's sidebar folder sort mode (`dashboard.folder_sort`), for every
 * surface that draws the folder tree outside the sidebar itself — the move-to
 * submenu, the new-chat-in-folder suggestion, the cron job form's folder picker.
 *
 * Read through the shared `['kirocrewConfig']` query rather than a dedicated
 * fetch: the sidebar already holds that query for its own settings, the sidebar
 * menu writes the mode through `api.patchConfig` and settles it back into the
 * same cache entry, so a picker opened right after a switch draws the new order
 * without a request of its own — and cannot lag behind the sidebar it sits next
 * to. The value is normalized by `readFolderSortMode`, so an older gateway
 * without the field, or a value this build does not know, reads as `custom`:
 * the stored order every earlier build drew.
 *
 * The read's failure travels with the mode (`error`) rather than being dropped
 * here: the sidebar reports the same failure on its own tree, and a host that
 * draws the tree elsewhere owes the person the same explanation, in the form
 * its surface allows (a block notice, an inline one, or the Radix menu pair).
 */
export function useFolderSortMode(): FolderSortModeRead {
  const { data, status, error } = useQuery<{ dashboard?: { folder_sort?: unknown } }>({
    queryKey: ['kirocrewConfig'],
    queryFn: () => api.kirocrewConfig(),
  })
  return {
    mode: readFolderSortMode(data?.dashboard?.folder_sort),
    error: status === 'error'
      ? (errMessage(error) || i18nT('components.errorBoundary.something_went_wrong'))
      : null,
  }
}
