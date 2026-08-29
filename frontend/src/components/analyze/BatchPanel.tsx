import { useCallback, useRef, useState } from 'react';

import { useBatchAnalyze } from '../../hooks/useBatchAnalyze';
import type { Analysis } from '../../lib/api/types';
import { QUALITY_TONE } from '../../lib/format';
import { Icon } from '../ui/Icon';
import { Badge, Button } from '../ui/primitives';

const ACCEPT = 'image/jpeg,image/png,image/webp,image/bmp,image/tiff';
const MAX_BYTES = 10 * 1024 * 1024;
const MAX_FILES = 10;

interface Props {
  disabled?: boolean;
  onOpen: (analysis: Analysis) => void;
  onDone: () => void;
}

/** Batch mode: pick several images, analyse them in one request, review a table. */
export function BatchPanel({ disabled, onOpen, onDone }: Props) {
  const { state, analyze, reset } = useBatchAnalyze();
  const inputRef = useRef<HTMLInputElement>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [notice, setNotice] = useState<string | null>(null);

  const addFiles = useCallback((incoming: FileList | null) => {
    if (!incoming) return;
    const picked = Array.from(incoming).filter(
      (f) => f.type.startsWith('image/') && f.size <= MAX_BYTES,
    );
    setNotice(
      picked.length < incoming.length ? 'Some files were skipped (not an image, or over 10 MB).' : null,
    );
    setFiles((prev) => {
      const seen = new Set(prev.map((f) => `${f.name}:${f.size}`));
      const merged = [...prev];
      for (const f of picked) {
        if (!seen.has(`${f.name}:${f.size}`)) merged.push(f);
      }
      return merged.slice(0, MAX_FILES);
    });
  }, []);

  const run = useCallback(async () => {
    if (!files.length) return;
    const result = await analyze(files);
    if (result) onDone();
  }, [files, analyze, onDone]);

  const startOver = useCallback(() => {
    setFiles([]);
    setNotice(null);
    reset();
  }, [reset]);

  const busy = state.status === 'analyzing';

  if (state.status === 'done') {
    const { result } = state;
    return (
      <div className="space-y-5">
        <div className="flex flex-wrap items-center gap-3">
          <p className="text-sm text-ink-soft">
            {`${result.succeeded} analysed` +
              (result.failed > 0 ? `, ${result.failed} failed` : '') +
              ` of ${result.total}`}
          </p>
          <Button variant="ghost" onClick={startOver}>
            <Icon name="refresh" size={14} /> New batch
          </Button>
        </div>

        <div className="clay-sm overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-[0.7rem] uppercase tracking-[0.14em] text-ink-faint">
              <tr>
                <th className="px-4 py-2.5 font-semibold">File</th>
                <th className="px-3 py-2.5 font-semibold">Score</th>
                <th className="px-3 py-2.5 font-semibold">Assessment</th>
                <th className="px-3 py-2.5 font-semibold">Issues</th>
                <th className="px-4 py-2.5" />
              </tr>
            </thead>
            <tbody className="divide-y divide-line/60">
              {result.items.map((item, i) => (
                <tr key={`${item.filename}-${i}`} className="align-middle">
                  <td className="max-w-[12rem] truncate px-4 py-2.5 text-ink">{item.filename}</td>
                  {item.ok && item.analysis ? (
                    <>
                      <td className="px-3 py-2.5 tabular-nums text-ink">
                        {item.analysis.quality_score != null
                          ? Math.round(item.analysis.quality_score)
                          : '—'}
                      </td>
                      <td className="px-3 py-2.5">
                        {item.analysis.quality_label && (
                          <Badge tone={QUALITY_TONE[item.analysis.quality_label]} className="px-1.5 py-0">
                            {item.analysis.quality_label}
                          </Badge>
                        )}
                      </td>
                      <td className="px-3 py-2.5 text-ink-soft">
                        {item.analysis.issues.length
                          ? item.analysis.issues.map((is) => is.type).join(', ')
                          : 'none'}
                      </td>
                      <td className="px-4 py-2.5 text-right">
                        <button
                          type="button"
                          onClick={() => item.analysis && onOpen(item.analysis)}
                          className="inline-flex items-center gap-1 text-xs font-medium text-brand hover:underline"
                        >
                          View <Icon name="chevronRight" size={13} />
                        </button>
                      </td>
                    </>
                  ) : (
                    <td colSpan={4} className="px-3 py-2.5 text-poor">
                      <Icon name="alert" size={13} className="mr-1 inline" />
                      {item.error?.message ?? 'Failed'}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <button
        type="button"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (!disabled) addFiles(e.dataTransfer.files);
        }}
        className="tray flex w-full flex-col items-center justify-center gap-3 px-6 py-10 text-center transition-all duration-300 disabled:cursor-not-allowed disabled:opacity-50"
      >
        <span className="clay-sm flex h-14 w-14 items-center justify-center text-brand">
          <Icon name="layers" size={26} />
        </span>
        <span className="font-display text-base font-semibold text-ink">
          Drop up to {MAX_FILES} images or click to browse
        </span>
        <span className="text-xs text-ink-faint">JPEG · PNG · WebP · BMP · TIFF · 10 MB each</span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        className="hidden"
        onChange={(e) => addFiles(e.target.files)}
      />

      {notice && (
        <p className="flex items-center gap-1.5 text-xs text-degraded">
          <Icon name="alert" size={13} /> {notice}
        </p>
      )}

      {files.length > 0 && (
        <ul className="clay-sm divide-y divide-line/60 text-sm">
          {files.map((f, i) => (
            <li key={`${f.name}-${i}`} className="flex items-center gap-3 px-4 py-2">
              <Icon name="file" size={14} className="shrink-0 text-ink-faint" />
              <span className="min-w-0 flex-1 truncate text-ink">{f.name}</span>
              <button
                type="button"
                onClick={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                className="shrink-0 text-ink-faint hover:text-poor"
                aria-label={`Remove ${f.name}`}
              >
                <Icon name="x" size={14} />
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="flex flex-wrap items-center gap-3">
        <Button onClick={run} disabled={busy || files.length === 0 || disabled}>
          <Icon name="sparkles" size={15} />
          {busy ? 'Analyzing…' : `Analyze ${files.length || ''} image${files.length === 1 ? '' : 's'}`}
        </Button>
        {files.length > 0 && !busy && (
          <Button variant="ghost" onClick={startOver}>
            Clear
          </Button>
        )}
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
  );
}
