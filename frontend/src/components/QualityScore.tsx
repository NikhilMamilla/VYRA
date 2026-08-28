import type { QualityLabel } from '../lib/api/types';
import { QUALITY_LABEL_STYLES, scoreColor, scoreTrackColor } from '../lib/format';

interface Props {
  score: number;
  label: QualityLabel;
}

/** Circular gauge for the 0–100 operational quality score. */
export function QualityScore({ score, label }: Props) {
  const radius = 52;
  const circumference = 2 * Math.PI * radius;
  const clamped = Math.max(0, Math.min(100, score));
  const offset = circumference * (1 - clamped / 100);

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative h-32 w-32">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 120 120">
          <circle
            cx="60"
            cy="60"
            r={radius}
            className="fill-none stroke-slate-200"
            strokeWidth="10"
          />
          <circle
            cx="60"
            cy="60"
            r={radius}
            className={`fill-none ${scoreTrackColor(clamped)} transition-[stroke-dashoffset] duration-700`}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold tabular-nums ${scoreColor(clamped)}`}>
            {Math.round(clamped)}
          </span>
          <span className="text-xs text-slate-400">/ 100</span>
        </div>
      </div>
      <span
        className={`rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wide ring-1 ring-inset ${QUALITY_LABEL_STYLES[label]}`}
      >
        {label}
      </span>
      <p className="max-w-[16rem] text-center text-xs text-slate-400">
        Operational quality score — derived from calibrated issue probabilities, not a
        perceptual rating.
      </p>
    </div>
  );
}
