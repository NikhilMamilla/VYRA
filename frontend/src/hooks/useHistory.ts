import { useCallback, useEffect, useState } from 'react';

import { listAnalyses } from '../lib/api/endpoints';
import type { Analysis } from '../lib/api/types';

interface State {
  items: Analysis[];
  total: number;
  loading: boolean;
  error: string | null;
}

const PAGE = 20;

/** Loads the analysis history and refreshes it on demand. */
export function useHistory() {
  const [state, setState] = useState<State>({
    items: [],
    total: 0,
    loading: true,
    error: null,
  });

  const refresh = useCallback((signal?: AbortSignal) => {
    setState((s) => ({ ...s, loading: true, error: null }));
    return listAnalyses({ limit: PAGE, offset: 0 }, signal)
      .then((page) => setState({ items: page.items, total: page.total, loading: false, error: null }))
      .catch((err: unknown) => {
        if (signal?.aborted) return;
        setState((s) => ({ ...s, loading: false, error: err instanceof Error ? err.message : String(err) }));
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    refresh(controller.signal);
    return () => controller.abort();
  }, [refresh]);

  return { ...state, refresh };
}
