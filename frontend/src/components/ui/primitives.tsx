import type { ButtonHTMLAttributes, HTMLAttributes, ReactNode } from 'react';

function cx(...parts: (string | false | null | undefined)[]) {
  return parts.filter(Boolean).join(' ');
}

/* -------- Container -------- */
export function Container({ className, ...rest }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cx('mx-auto w-full max-w-6xl px-5 sm:px-6', className)} {...rest} />;
}

/* -------- Card (surface variants) -------- */
type Surface = 'glass' | 'clay' | 'clay-sm' | 'neu' | 'plain';
const SURFACE: Record<Surface, string> = {
  glass: 'glass-panel',
  clay: 'clay',
  'clay-sm': 'clay-sm',
  neu: 'neu rounded-2xl',
  plain: 'rounded-2xl border border-line bg-surface-solid',
};

export function Card({
  as: As = 'div',
  surface = 'glass',
  className,
  children,
  ...rest
}: HTMLAttributes<HTMLElement> & { as?: 'div' | 'section' | 'article' | 'li'; surface?: Surface }) {
  return (
    <As className={cx(SURFACE[surface], className)} {...rest}>
      {children}
    </As>
  );
}

/* -------- Button -------- */
type Variant = 'primary' | 'ghost' | 'soft';
const BUTTON: Record<Variant, string> = {
  primary:
    'text-white bg-brand shadow-glow hover:brightness-110 active:brightness-95 active:translate-y-px',
  ghost:
    'text-ink border border-line bg-surface hover:bg-bg-alt backdrop-blur-sm',
  soft: 'neu-pressable text-ink',
};

export function Button({
  variant = 'primary',
  className,
  children,
  ...rest
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={cx(
        'inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold',
        'transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-55',
        BUTTON[variant],
        className,
      )}
      {...rest}
    >
      {children}
    </button>
  );
}

/* -------- Badge / Pill -------- */
export function Badge({
  tone = 'neutral',
  className,
  title,
  children,
}: {
  tone?: 'neutral' | 'brand' | 'good' | 'ok' | 'degraded' | 'poor' | 'real' | 'synth' | 'screen';
  className?: string;
  title?: string;
  children: ReactNode;
}) {
  const tones: Record<string, string> = {
    neutral: 'bg-ink/5 text-ink-soft ring-ink/10',
    brand: 'bg-brand/10 text-brand ring-brand/20',
    good: 'bg-good/12 text-good ring-good/25',
    ok: 'bg-ok/14 text-ok ring-ok/25',
    degraded: 'bg-degraded/14 text-degraded ring-degraded/25',
    poor: 'bg-poor/12 text-poor ring-poor/25',
    real: 'bg-tier-real/12 text-tier-real ring-tier-real/25',
    synth: 'bg-tier-synth/12 text-tier-synth ring-tier-synth/25',
    screen: 'bg-tier-screen/12 text-tier-screen ring-tier-screen/25',
  };
  return (
    <span
      title={title}
      className={cx(
        'inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset',
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/* -------- Section heading -------- */
export function SectionHeading({
  eyebrow,
  title,
  lead,
  id,
}: {
  eyebrow: string;
  title: string;
  lead?: string;
  id?: string;
}) {
  return (
    <div id={id} className="mb-10 max-w-2xl scroll-mt-24">
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.2em] text-brand">{eyebrow}</p>
      <h2 className="text-3xl font-bold text-ink sm:text-4xl">{title}</h2>
      {lead && <p className="mt-3 text-base leading-relaxed text-ink-soft">{lead}</p>}
    </div>
  );
}
