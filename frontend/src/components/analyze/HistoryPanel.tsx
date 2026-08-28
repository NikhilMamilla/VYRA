import type { Analysis } from '../../lib/api/types';
import { QUALITY_TONE, relativeTime } from '../../lib/format';
import { Icon } from '../ui/Icon';
import { Badge } from '../ui/primitives';

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
    <section className="glass-panel overflow-hidden">
      <header className="flex items-center justify-between border-b border-line/70 px-4 py-3">
        <h3 className="flex items-center gap-2 font-display text-sm font-semibold text-ink">
          <Icon name="history" size={15} /> History
        </h3>
        <span className="text-xs tabular-nums text-ink-faint">{total}</span>
      </header>

      {loading && <p className="px-4 py-8 text-sm text-ink-faint">Loading…</p>}
      {error && <p className="px-4 py-8 text-sm text-poor">{error}</p>}
      {!loading && !error && items.length === 0 && (
        <p className="px-4 py-8 text-sm text-ink-faint">No analyses yet — upload an image to start.</p>
      )}

      <ul className="max-h-[30rem] divide-y divide-line/60 overflow-y-auto">
        {items.map((a) => (
          <li key={a.id}>
            <button
              type="button"
              onClick={() => onSelect(a)}
              className={`flex w-full items-center gap-3 px-4 py-3 text-left transition-colors hover:bg-ink/[0.03] ${
                selectedId === a.id ? 'bg-brand/[0.06]' : ''
              }`}
            >
              <span className="clay-sm flex h-11 w-12 shrink-0 items-center justify-center font-display text-sm font-bold tabular-nums text-ink">
                {a.quality_score != null ? Math.round(a.quality_score) : '—'}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-ink">{a.image.filename}</span>
                <span className="mt-0.5 flex items-center gap-1.5 text-xs text-ink-faint">
                  {relativeTime(a.created_at)}
                  {a.quality_label && (
                    <Badge tone={QUALITY_TONE[a.quality_label]} className="px-1.5 py-0">
                      {a.quality_label}
                    </Badge>
                  )}
                </span>
              </span>
              <Icon name="chevronRight" size={14} className="shrink-0 text-ink-faint" />
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
