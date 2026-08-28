import type { Analysis } from '../lib/api/types';
import { QUALITY_LABEL_STYLES, formatDateTime } from '../lib/format';

interface Props {
  items: Analysis[];
  total: number;
  loading: boolean;
  error: string | null;
  selectedId: string | null;
  onSelect: (a: Analysis) => void;
}

export function HistoryPanel({ items, total, loading, error, selectedId, onSelect }: Props) {
  return (
    <section className="rounded-xl border border-slate-200 bg-white">
      <header className="flex items-baseline justify-between border-b border-slate-100 px-4 py-3">
        <h2 className="text-sm font-semibold text-slate-700">History</h2>
        <span className="text-xs text-slate-400">{total} total</span>
      </header>

      {loading && <p className="px-4 py-6 text-sm text-slate-400">Loading…</p>}
      {error && <p className="px-4 py-6 text-sm text-red-600">{error}</p>}
      {!loading && !error && items.length === 0 && (
        <p className="px-4 py-6 text-sm text-slate-400">No analyses yet.</p>
      )}

      <ul className="max-h-[28rem] divide-y divide-slate-100 overflow-y-auto">
        {items.map((a) => (
          <li key={a.id}>
            <button
              type="button"
              onClick={() => onSelect(a)}
              className={`flex w-full items-center gap-3 px-4 py-3 text-left hover:bg-slate-50 ${
                selectedId === a.id ? 'bg-slate-50' : ''
              }`}
            >
              <span
                className={`inline-flex h-9 w-11 shrink-0 items-center justify-center rounded text-xs font-semibold ${
                  a.quality_label ? QUALITY_LABEL_STYLES[a.quality_label] : 'bg-slate-100 text-slate-500'
                }`}
              >
                {a.quality_score != null ? Math.round(a.quality_score) : '—'}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-slate-700">
                  {a.image.filename}
                </span>
                <span className="block text-xs text-slate-400">
                  {formatDateTime(a.created_at)}
                  {a.issues.length > 0 && ` · ${a.issues.map((i) => i.type).join(', ')}`}
                </span>
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
