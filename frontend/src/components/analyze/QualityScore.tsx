import { useEffect, useState } from 'react';

import type { QualityLabel } from '../../lib/api/types';
import { QUALITY_TONE, scoreVar } from '../../lib/format';
import { Badge } from '../ui/primitives';

interface Props {
  score: number;
  label: QualityLabel;
}

/**
 * The score dial: an SVG arc rides in a neumorphic recess (extruded from the
 * surface), a skeuomorphic tick ring around it, the number in a soft inset well.
 * The number is rendered immediately; the arc sweeps in via a CSS transition.
 */
export function QualityScore({ score, label }: Props) {
  const clamped = Math.max(0, Math.min(100, score));
  const [progress, setProgress] = useState(0);

  // Kick the arc from 0 -> value on mount so the CSS transition plays.
  useEffect(() => {
    const id = requestAnimationFrame(() => setProgress(clamped));
    return () => cancelAnimationFrame(id);
  }, [clamped]);

  const r = 76;
  const circ = 2 * Math.PI * r;
  const arc = 0.75; // 270° sweep, gap at the bottom
  const dash = circ * arc;
  const offset = dash * (1 - progress / 100);

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="neu relative grid h-52 w-52 place-items-center rounded-full">
        <svg viewBox="0 0 200 200" className="absolute inset-0 h-full w-full -rotate-[135deg]">
          {Array.from({ length: 28 }).map((_, i) => {
            const a = (i / 28) * arc * 2 * Math.PI;
            const outer = 92;
            const inner = i % 7 === 0 ? 83 : 87;
            return (
              <line
                key={i}
                x1={100 + Math.cos(a) * outer}
                y1={100 + Math.sin(a) * outer}
                x2={100 + Math.cos(a) * inner}
                y2={100 + Math.sin(a) * inner}
                stroke="rgb(var(--c-ink) / 0.14)"
                strokeWidth={i % 7 === 0 ? 2 : 1}
                strokeLinecap="round"
              />
            );
          })}
          <circle
            cx="100"
            cy="100"
            r={r}
            fill="none"
            stroke="rgb(var(--c-ink) / 0.08)"
            strokeWidth="12"
            strokeDasharray={`${dash} ${circ}`}
            strokeLinecap="round"
          />
          <circle
            cx="100"
            cy="100"
            r={r}
            fill="none"
            stroke={scoreVar(clamped)}
            strokeWidth="12"
            strokeDasharray={`${dash} ${circ}`}
            strokeDashoffset={offset}
            strokeLinecap="round"
            style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.22,1,0.36,1), stroke 0.4s' }}
          />
        </svg>
        <div className="neu-inset flex h-32 w-32 flex-col items-center justify-center rounded-full">
          <span
            className="font-display text-5xl font-bold tabular-nums leading-none"
            style={{ color: scoreVar(clamped) }}
          >
            {Math.round(clamped)}
          </span>
          <span className="mt-1 text-[0.7rem] font-medium uppercase tracking-wider text-ink-faint">
            / 100
          </span>
        </div>
      </div>

      <Badge tone={QUALITY_TONE[label]} className="px-3 py-1 text-sm font-semibold tracking-wide">
        {label}
      </Badge>
      <p className="max-w-[15rem] text-center text-xs leading-relaxed text-ink-faint">
        Operational quality score — from calibrated issue probabilities, not a perceptual rating.
      </p>
    </div>
  );
}
