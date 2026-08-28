import type { Explanation } from '../lib/api/types';
import { humaniseFeature } from '../lib/format';

export function ExplanationPanel({ explanation }: { explanation: Explanation }) {
  const evidence = explanation.evidence ?? [];
  const defect = explanation.potential_defect;

  return (
    <div className="space-y-3 text-sm">
      {explanation.summary && <p className="text-slate-700">{explanation.summary}</p>}

      {evidence.length > 0 && (
        <div>
          <h4 className="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-400">
            Why these were flagged
          </h4>
          <ul className="space-y-1">
            {evidence.map((e, i) => (
              <li key={`${e.feature}-${i}`} className="flex items-baseline justify-between gap-3">
                <span className="text-slate-600">{humaniseFeature(e.feature)}</span>
                <span className="tabular-nums text-slate-400">
                  {e.value.toLocaleString(undefined, { maximumFractionDigits: 3 })}
                  <span className="ml-2 text-slate-300">{e.direction.replace(/_/g, ' ')}</span>
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {defect?.flagged && (
        <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600">
          <span className="font-medium text-slate-700">Potential visual defect: </span>
          {defect.note}
          {defect.evidence.length > 0 && (
            <span className="block mt-1 text-slate-400">
              local anomaly in {defect.evidence.map((f) => humaniseFeature(f.feature)).join(', ')}
            </span>
          )}
        </div>
      )}

      {explanation.feature_version && (
        <p className="text-xs text-slate-300">
          feature engine {explanation.feature_version}
          {explanation.timings_ms?.total != null && ` · ${explanation.timings_ms.total} ms`}
        </p>
      )}
    </div>
  );
}
