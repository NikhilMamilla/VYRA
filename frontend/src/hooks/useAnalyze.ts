import { useCallback, useState } from 'react';

import { ApiError, NetworkError } from '../lib/api/client';
import { createAnalysis } from '../lib/api/endpoints';
import type { Analysis } from '../lib/api/types';

type State =
  | { status: 'idle' }
  | { status: 'analyzing' }
  | { status: 'done'; analysis: Analysis }
  | { status: 'error'; message: string; code: string };

const FRIENDLY: Record<string, string> = {
  payload_too_large: 'That file is too large. The limit is 10 MB.',
  unsupported_media_type: 'That file type is not supported. Use JPEG, PNG, WebP, BMP or TIFF.',
  invalid_image: 'That file could not be read as an image.',
  not_implemented: 'Image analysis is not available on this server (no model is loaded).',
};

/** Drives one upload → analyze request and exposes its state. */
export function useAnalyze() {
  const [state, setState] = useState<State>({ status: 'idle' });

  const analyze = useCallback(async (file: File) => {
    setState({ status: 'analyzing' });
    try {
      const analysis = await createAnalysis(file);
      setState({ status: 'done', analysis });
      return analysis;
    } catch (err) {
      if (err instanceof ApiError) {
        setState({
          status: 'error',
          code: err.code,
          message: FRIENDLY[err.code] ?? err.message,
        });
      } else if (err instanceof NetworkError) {
        setState({
          status: 'error',
          code: 'network',
          message: 'Could not reach the VYRA API. Is the backend running?',
        });
      } else {
        setState({ status: 'error', code: 'unknown', message: String(err) });
      }
      return null;
    }
  }, []);

  const reset = useCallback(() => setState({ status: 'idle' }), []);

  return { state, analyze, reset };
}
