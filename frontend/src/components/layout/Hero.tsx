import { MODEL } from '../../content/model';
import { Icon } from '../ui/Icon';
import { Container } from '../ui/primitives';

const CHIPS = [
  ['real-world F1', '0.61 blur'],
  ['features', `${MODEL.features} CV`],
  ['calibration', 'isotonic'],
  ['deploy', 'Docker'],
];

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden">
      {/* soft brand aura */}
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/2 h-[38rem] w-[38rem] -translate-x-1/2 rounded-full bg-brand-soft blur-3xl"
      />
      <Container className="relative py-16 sm:py-24">
        <div className="mx-auto max-w-3xl text-center">
          <span className="glass mb-6 inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium text-ink-soft">
            <Icon name="sparkles" size={13} className="text-brand" />
            Classical CV + calibrated ML · no external AI APIs
          </span>
          <h1 className="font-display text-4xl font-bold leading-[1.08] tracking-tight text-ink sm:text-6xl">
            Know an image's{' '}
            <span className="gradient-text">quality</span> before it costs you.
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-base leading-relaxed text-ink-soft sm:text-lg">
            VYRA scores visual quality, flags blur, exposure, noise, corruption and potential
            defects — with per-issue confidence, severity, the statistics behind the call, and an
            honest note on how far each detector has actually been validated.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <a
              href="#analyze"
              className="inline-flex items-center gap-2 rounded-xl bg-brand px-5 py-3 text-sm font-semibold text-white shadow-glow transition hover:brightness-110 active:translate-y-px"
            >
              Analyze an image <Icon name="arrowDown" size={15} />
            </a>
            <a
              href="#how"
              className="neu-pressable inline-flex items-center gap-2 rounded-xl px-5 py-3 text-sm font-semibold text-ink"
            >
              How it works
            </a>
          </div>

          <dl className="mx-auto mt-12 grid max-w-lg grid-cols-2 gap-3 sm:grid-cols-4">
            {CHIPS.map(([k, v]) => (
              <div key={k} className="clay-sm px-3 py-2.5 text-center">
                <dt className="text-[0.65rem] uppercase tracking-wide text-ink-faint">{k}</dt>
                <dd className="mt-0.5 font-display text-sm font-semibold text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
      </Container>
    </section>
  );
}
