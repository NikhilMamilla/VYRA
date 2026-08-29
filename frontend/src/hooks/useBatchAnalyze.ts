import { useCallback, useState } from 'react';

import { ApiError, NetworkError } from '../lib/api/client';
import { createAnalysesBatch } from '../lib/api/endpoints';
import type { BatchAnalysisResponse } from '../lib/api/types';

type State =
  | { status: 'idle' }
  | { status: 'analyzing' }
  | { status: 'done'; result: BatchAnalysisResponse }
  | { status: 'error'; message: string };

const FRIENDLY: Record<string, string> = {
  payload_too_large: 'Too many images in one batch. The limit is shown below.',
  not_implemented: 'Image analysis is not available on this server (no model is loaded).',
};

/** Drives one multi-image upload → analyze request and exposes its state. */
export function useBatchAnalyze() {
  const [state, setState] = useState<State>({ status: 'idle' });

  const analyze = useCallback(async (files: File[]) => {
    setState({ status: 'analyzing' });
    try {
      const result = await createAnalysesBatch(files);
      setState({ status: 'done', result });
      return result;
    } catch (err) {
      if (err instanceof ApiError) {
        setState({ status: 'error', message: FRIENDLY[err.code] ?? err.message });
      } else if (err instanceof NetworkError) {
        setState({ status: 'error', message: 'Could not reach the VYRA API. Is the backend running?' });
      } else {
        setState({ status: 'error', message: String(err) });
      }
      return null;
    }
  }, []);

  const reset = useCallback(() => setState({ status: 'idle' }), []);

  return { state, analyze, reset };
}
