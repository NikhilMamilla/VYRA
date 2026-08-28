import { STATISTIC_LABELS } from '../lib/format';

export function StatisticsGrid({ metrics }: { metrics: Record<string, number> }) {
  const entries = Object.entries(metrics);
  if (entries.length === 0) return null;

  return (
    <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
      {entries.map(([key, value]) => (
        <div key={key} className="rounded-lg bg-slate-50 px-3 py-2">
          <dt className="text-xs text-slate-500">{STATISTIC_LABELS[key] ?? key}</dt>
          <dd className="mt-0.5 font-medium tabular-nums text-slate-800">
            {Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : '—'}
          </dd>
        </div>
      ))}
    </dl>
  );
}
