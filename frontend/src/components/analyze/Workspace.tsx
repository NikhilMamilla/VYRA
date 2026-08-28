import { useCallback, useEffect, useRef, useState } from 'react';

import { useAnalyze } from '../../hooks/useAnalyze';
import { useHistory } from '../../hooks/useHistory';
import type { Analysis } from '../../lib/api/types';
import { Icon } from '../ui/Icon';
import { Button, Card, Container } from '../ui/primitives';
import { AnalysisResult } from './AnalysisResult';
import { HistoryPanel } from './HistoryPanel';
import { UploadDropzone } from './UploadDropzone';

export function Workspace({ analyzerReady }: { analyzerReady: boolean | null }) {
  const { state, analyze, reset } = useAnalyze();
  const history = useHistory();

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selected, setSelected] = useState<Analysis | null>(null);
  const previewRef = useRef<string | null>(null);

  useEffect(
    () => () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    },
    [],
  );

  const chooseFile = useCallback(
    (f: File) => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
      const url = URL.createObjectURL(f);
      previewRef.current = url;
      setPreviewUrl(url);
      setFile(f);
      setSelected(null);
      reset();
    },
    [reset],
  );

  const runAnalysis = useCallback(async () => {
    if (!file) return;
    const result = await analyze(file);
    if (result) {
      setSelected(null);
      history.refresh();
    }
  }, [file, analyze, history]);

  const startOver = useCallback(() => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    previewRef.current = null;
    setPreviewUrl(null);
    setFile(null);
    setSelected(null);
    reset();
  }, [reset]);

  const shownAnalysis = selected ?? (state.status === 'done' ? state.analysis : null);
  const shownPreview = selected ? null : previewUrl;
  const busy = state.status === 'analyzing';

  return (
    <Container id="analyze" className="scroll-mt-24 py-10 sm:py-16">
      <div className="mb-8 flex flex-wrap items-end justify-between gap-3">
        <div>
          <p className="mb-1 text-xs font-semibold uppercase tracking-[0.2em] text-brand">Workspace</p>
          <h2 className="font-display text-3xl font-bold text-ink">Analyze an image</h2>
        </div>
        {analyzerReady === false && (
          <span className="flex items-center gap-1.5 text-sm text-degraded">
            <Icon name="alert" size={15} /> model not loaded on the server
          </span>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
        <Card surface="glass" className="p-5 sm:p-7">
          {!file && !selected && (
            <UploadDropzone disabled={analyzerReady === false} onFile={chooseFile} />
          )}

          {file && !shownAnalysis && (
            <div className="space-y-5">
              {previewUrl && (
                <figure className="clay-sm overflow-hidden p-2.5">
                  <img
                    src={previewUrl}
                    alt={file.name}
                    className="mx-auto max-h-80 w-auto rounded-[0.9rem] object-contain"
                  />
                </figure>
              )}
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={runAnalysis} disabled={busy}>
                  <Icon name="sparkles" size={16} />
                  {busy ? 'Analyzing…' : 'Analyze image'}
                </Button>
                <Button variant="ghost" onClick={startOver} disabled={busy}>
                  Choose another
                </Button>
                <span className="text-xs text-ink-faint">{file.name}</span>
              </div>
              {busy && (
                <div className="neu-inset relative h-2 overflow-hidden rounded-full">
                  <div className="absolute inset-y-0 left-0 w-1/3 animate-[shimmer_1.1s_infinite] rounded-full bg-brand" />
                </div>
              )}
              {state.status === 'error' && (
                <p className="flex items-center gap-2 rounded-xl bg-poor/8 px-4 py-3 text-sm text-poor">
                  <Icon name="alert" size={16} /> {state.message}
                </p>
              )}
            </div>
          )}

          {shownAnalysis && (
            <div className="space-y-5">
              <AnalysisResult analysis={shownAnalysis} previewUrl={shownPreview} />
              <Button onClick={startOver}>
                <Icon name="refresh" size={15} /> Analyze another image
              </Button>
            </div>
          )}
        </Card>

        <aside className="lg:sticky lg:top-24 lg:self-start">
          <HistoryPanel
            items={history.items}
            total={history.total}
            loading={history.loading}
            error={history.error}
            selectedId={selected?.id ?? null}
            onSelect={(a) => {
              setSelected(a);
              setFile(null);
            }}
          />
        </aside>
      </div>
    </Container>
  );
}
