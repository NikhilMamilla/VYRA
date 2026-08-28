import type { Issue } from '../lib/api/types';
import { ISSUE_LABELS, SEVERITY_STYLES, VALIDATION_NOTE } from '../lib/format';

const VALIDATION_BADGE: Record<string, string> = {
  'real-world': 'bg-emerald-50 text-emerald-700 ring-emerald-600/20',
  'synthetic-only': 'bg-sky-50 text-sky-700 ring-sky-600/20',
  screening: 'bg-slate-100 text-slate-600 ring-slate-500/20',
};

export function IssueList({ issues }: { issues: Issue[] }) {
  if (issues.length === 0) {
    return (
      <p className="rounded-lg bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
        No quality issues were flagged for this image.
      </p>
    );
  }

  return (
    <ul className="space-y-2">
      {issues.map((issue) => (
        <li
          key={issue.type}
          className="rounded-lg border border-slate-200 bg-white p-3"
        >
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium text-slate-900">
              {ISSUE_LABELS[issue.type] ?? issue.type}
            </span>
            <span
              className={`rounded px-1.5 py-0.5 text-xs font-medium ${SEVERITY_STYLES[issue.severity]}`}
            >
              {issue.severity} severity
            </span>
            {issue.validation && (
              <span
                className={`rounded px-1.5 py-0.5 text-xs font-medium ring-1 ring-inset ${VALIDATION_BADGE[issue.validation]}`}
                title={VALIDATION_NOTE[issue.validation]}
              >
                {issue.validation}
              </span>
            )}
            <span className="ml-auto text-sm tabular-nums text-slate-500">
              {(issue.confidence * 100).toFixed(0)}% confidence
            </span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-slate-100">
            <div
              className="h-full rounded-full bg-slate-400"
              style={{ width: `${Math.round(issue.confidence * 100)}%` }}
            />
          </div>
          {issue.detail && <p className="mt-2 text-xs text-slate-500">{issue.detail}</p>}
        </li>
      ))}
    </ul>
  );
}
