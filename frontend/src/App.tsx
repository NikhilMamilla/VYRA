import { useCallback, useEffect, useRef, useState } from 'react';

import { AnalysisResult } from './components/AnalysisResult';
import { HistoryPanel } from './components/HistoryPanel';
import { UploadDropzone } from './components/UploadDropzone';
import { useAnalyze } from './hooks/useAnalyze';
import { useBackendHealth } from './hooks/useBackendHealth';
import { useHistory } from './hooks/useHistory';
import type { Analysis } from './lib/api/types';

export default function App() {
  const health = useBackendHealth();
  const { state, analyze, reset } = useAnalyze();
  const history = useHistory();

  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selected, setSelected] = useState<Analysis | null>(null);
  const previewRef = useRef<string | null>(null);

  useEffect(() => {
    return () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current);
    };
  }, []);

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

  const analyzerReady = health.status === 'success' && health.health.components.analyzer?.status === 'ok';
  const shownAnalysis = selected ?? (state.status === 'done' ? state.analysis : null);
  const shownPreview = selected ? null : previewUrl;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      <div className="mx-auto w-full max-w-6xl px-4 py-8">
        <header className="mb-8 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">VYRA</h1>
            <p className="text-sm text-slate-500">AI-powered image quality &amp; defect detection</p>
          </div>
          <HealthBadge health={health} />
        </header>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_20rem]">
          <main className="space-y-6">
            <section className="rounded-xl border border-slate-200 bg-white p-6">
              {!file && !selected && (
                <>
                  <UploadDropzone disabled={!analyzerReady} onFile={chooseFile} />
                  {!analyzerReady && health.status === 'success' && (
                    <p className="mt-3 text-sm text-amber-700">
                      The analysis model is not loaded on the server, so uploads cannot be
                      analyzed right now.
                    </p>
                  )}
                </>
              )}

              {file && state.status !== 'done' && (
                <div className="space-y-4">
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt={file.name}
                      className="mx-auto max-h-80 w-auto rounded-lg border border-slate-200 object-contain"
                    />
                  )}
                  <div className="flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      onClick={runAnalysis}
                      disabled={state.status === 'analyzing'}
                      className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700 disabled:opacity-60"
                    >
                      {state.status === 'analyzing' ? 'Analyzing…' : 'Analyze image'}
                    </button>
                    <button
                      type="button"
                      onClick={startOver}
                      className="rounded-md border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                    >
                      Choose a different image
                    </button>
                    <span className="text-xs text-slate-400">{file.name}</span>
                  </div>
                  {state.status === 'analyzing' && (
                    <div className="h-1 w-full overflow-hidden rounded bg-slate-100">
                      <div className="h-full w-1/3 animate-pulse rounded bg-slate-400" />
                    </div>
                  )}
                  {state.status === 'error' && (
                    <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
                      {state.message}
                    </p>
                  )}
                </div>
              )}

              {shownAnalysis && (
                <div className="space-y-4">
                  <AnalysisResult analysis={shownAnalysis} previewUrl={shownPreview} />
                  <button
                    type="button"
                    onClick={startOver}
                    className="rounded-md bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
                  >
                    Analyze another image
                  </button>
                </div>
              )}
            </section>
          </main>

          <aside>
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
      </div>
    </div>
  );
}

function HealthBadge({ health }: { health: ReturnType<typeof useBackendHealth> }) {
  if (health.status === 'loading') {
    return <span className="text-xs text-slate-400">checking server…</span>;
  }
  if (health.status === 'error') {
    return (
      <span className="rounded-full bg-red-100 px-3 py-1 text-xs font-medium text-red-800">
        API unreachable
      </span>
    );
  }
  const analyzer = health.health.components.analyzer;
  const ok = analyzer?.status === 'ok';
  return (
    <span
      className={`rounded-full px-3 py-1 text-xs font-medium ${
        ok ? 'bg-emerald-100 text-emerald-800' : 'bg-amber-100 text-amber-900'
      }`}
      title={analyzer?.detail ?? undefined}
    >
      {ok ? `model ${health.health.analyzer_model_version}` : 'model not loaded'}
    </span>
  );
}
