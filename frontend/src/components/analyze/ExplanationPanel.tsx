import type { Explanation } from '../../lib/api/types';
import { humaniseFeature } from '../../lib/format';
import { Icon } from '../ui/Icon';

export function ExplanationPanel({ explanation }: { explanation: Explanation }) {
  const evidence = explanation.evidence ?? [];
  const defect = explanation.potential_defect;

  return (
    <div className="space-y-4 text-sm">
      {explanation.summary && (
        <p className="leading-relaxed text-ink-soft">{explanation.summary}</p>
      )}

      {evidence.length > 0 && (
        <div>
          <h4 className="mb-2 flex items-center gap-1.5 text-[0.7rem] font-semibold uppercase tracking-wider text-ink-faint">
            <Icon name="scan" size={13} /> Why these were flagged
          </h4>
          <ul className="divide-y divide-line/70 rounded-xl border border-line/70">
            {evidence.map((e, i) => (
              <li key={`${e.feature}-${i}`} className="flex items-baseline justify-between gap-3 px-3 py-2">
                <span className="text-ink-soft">{humaniseFeature(e.feature)}</span>
                <span className="flex items-baseline gap-2 tabular-nums text-ink-faint">
                  <span className="font-medium text-ink">
                    {e.value.toLocaleString(undefined, { maximumFractionDigits: 3 })}
                  </span>
                  <span className="hidden text-[0.7rem] sm:inline">
                    {e.direction.replace(/_/g, ' ')}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {defect?.flagged && (
        <div className="rounded-xl bg-poor/8 p-3 text-xs leading-relaxed text-ink-soft">
          <span className="font-semibold text-poor">Potential visual defect · </span>
          {defect.note}
          {defect.evidence.length > 0 && (
            <span className="mt-1 block text-ink-faint">
              local anomaly in {defect.evidence.map((f) => humaniseFeature(f.feature)).join(', ')}
            </span>
          )}
        </div>
      )}

      {explanation.feature_version && (
        <p className="flex items-center gap-1.5 text-[0.7rem] text-ink-faint">
          <Icon name="cpu" size={12} />
          feature engine {explanation.feature_version}
          {explanation.timings_ms?.total != null && ` · ${explanation.timings_ms.total} ms`}
        </p>
      )}
    </div>
  );
}
