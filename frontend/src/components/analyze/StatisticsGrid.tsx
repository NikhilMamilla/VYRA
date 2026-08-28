import { STATISTIC_LABELS } from '../../lib/format';

export function StatisticsGrid({ metrics }: { metrics: Record<string, number> }) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) return null;

  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3">
      {entries.map(([key, value]) => (
        <div key={key} className="clay-sm px-3.5 py-3">
          <dt className="text-[0.7rem] font-medium uppercase tracking-wide text-ink-faint">
            {STATISTIC_LABELS[key] ?? key}
          </dt>
          <dd className="mt-1 font-display text-lg font-semibold tabular-nums text-ink">
            {Number.isFinite(value)
              ? value.toLocaleString(undefined, { maximumFractionDigits: 3 })
              : '—'}
          </dd>
        </div>
      ))}
    </dl>
  );
}
