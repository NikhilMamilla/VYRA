import type { Issue } from '../../lib/api/types';
import {
  ISSUE_LABELS,
  SEVERITY_LABEL,
  VALIDATION_LABEL,
  VALIDATION_NOTE,
  VALIDATION_TONE,
  severityTone,
} from '../../lib/format';
import { Icon } from '../ui/Icon';
import { Badge } from '../ui/primitives';

export function IssueList({ issues }: { issues: Issue[] }) {
  if (issues.length === 0) {
    return (
      <div className="clay-sm flex items-center gap-3 px-4 py-4 text-sm text-good">
        <span className="clay-sm flex h-9 w-9 items-center justify-center">
          <Icon name="check" size={18} />
        </span>
        No quality issues were flagged for this image.
      </div>
    );
  }

  return (
    <ul className="grid gap-3">
      {issues.map((issue) => (
        <li key={issue.type} className="clay-sm p-4">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
            <span className="font-display text-[0.95rem] font-semibold text-ink">
              {ISSUE_LABELS[issue.type] ?? issue.type}
            </span>
            <Badge tone={severityTone(issue.severity)}>{SEVERITY_LABEL[issue.severity]}</Badge>
            {issue.validation && (
              <Badge
                tone={VALIDATION_TONE[issue.validation]}
                className="cursor-help"
                title={VALIDATION_NOTE[issue.validation]}
              >
                {VALIDATION_LABEL[issue.validation]}
              </Badge>
            )}
            <span className="ml-auto text-sm font-semibold tabular-nums text-ink-soft">
              {(issue.confidence * 100).toFixed(0)}%
            </span>
          </div>

          <div className="neu-inset mt-2.5 h-2 overflow-hidden rounded-full">
            <div
              className="h-full rounded-full bg-brand transition-[width] duration-700"
              style={{ width: `${Math.max(4, Math.round(issue.confidence * 100))}%` }}
            />
          </div>

          {issue.detail && <p className="mt-2 text-xs leading-relaxed text-ink-faint">{issue.detail}</p>}
        </li>
      ))}
    </ul>
  );
}
