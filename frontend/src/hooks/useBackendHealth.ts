import { useCallback, useEffect, useState } from 'react';

import { getHealth } from '../lib/api/endpoints';
import type { Health } from '../lib/api/types';

type State =
  | { status: 'loading'; health: null; error: null }
  | { status: 'success'; health: Health; error: null }
  | { status: 'error'; health: null; error: Error };

const INITIAL: State = { status: 'loading', health: null, error: null };

/** Polls the backend `/health` endpoint once, with a manual retry. */
export function useBackendHealth() {
  const [state, setState] = useState<State>(INITIAL);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setState(INITIAL);

    getHealth(controller.signal)
      .then((health) => setState({ status: 'success', health, error: null }))
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setState({
          status: 'error',
          health: null,
          error: error instanceof Error ? error : new Error(String(error)),
        });
      });

    return () => controller.abort();
  }, [attempt]);

  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  return { ...state, retry };
}
