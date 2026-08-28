import type { PotentialDefect } from '../../lib/api/types';

interface Props {
  src: string;
  alt: string;
  defect?: PotentialDefect | null;
}

/** Image preview framed like a print; defect region drawn as a highlighter stroke. */
export function ImageCanvas({ src, alt, defect }: Props) {
  const region = defect?.flagged ? defect.region : null;

  return (
    <figure className="clay-sm overflow-hidden p-2.5">
      <div className="relative overflow-hidden rounded-[0.9rem] bg-bg-alt">
        <img src={src} alt={alt} className="mx-auto max-h-[24rem] w-auto object-contain" />
        {region && (
          <>
            <span
              className="pointer-events-none absolute rounded-[3px] border-2 border-poor/90 mix-blend-multiply dark:mix-blend-screen"
              style={{
                left: `${region[0] * 100}%`,
                top: `${region[1] * 100}%`,
                width: `${region[2] * 100}%`,
                height: `${region[3] * 100}%`,
                background: 'rgb(var(--c-poor) / 0.22)',
                boxShadow: '0 2px 10px rgb(var(--c-poor) / 0.4)',
                transform: 'rotate(-0.6deg)',
              }}
            />
            <span
              className="pointer-events-none absolute rounded-full bg-poor px-2 py-0.5 text-[0.65rem] font-semibold text-white"
              style={{
                left: `${region[0] * 100}%`,
                top: `calc(${region[1] * 100}% - 1.4rem)`,
              }}
            >
              potential defect
            </span>
          </>
        )}
      </div>
    </figure>
  );
}
