import type { PotentialDefect } from '../lib/api/types';

interface Props {
  src: string;
  alt: string;
  defect?: PotentialDefect | null;
}

/** Image preview with an optional highlighted defect region. */
export function ImageCanvas({ src, alt, defect }: Props) {
  const region = defect?.flagged ? defect.region : null;

  return (
    <div className="relative overflow-hidden rounded-xl border border-slate-200 bg-slate-100">
      <img src={src} alt={alt} className="mx-auto max-h-[22rem] w-auto object-contain" />
      {region && (
        <div
          className="pointer-events-none absolute border-2 border-red-500/90 bg-red-500/10"
          style={{
            left: `${region[0] * 100}%`,
            top: `${region[1] * 100}%`,
            width: `${region[2] * 100}%`,
            height: `${region[3] * 100}%`,
          }}
          title="Potential visual defect (screening)"
        />
      )}
    </div>
  );
}
