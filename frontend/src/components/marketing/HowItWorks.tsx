import { PIPELINE } from '../../content/model';
import { Icon } from '../ui/Icon';
import type { IconName } from '../ui/Icon';
import { Container, SectionHeading } from '../ui/primitives';

export function HowItWorks() {
  return (
    <section className="bg-bg-alt/60 py-16 sm:py-24">
      <Container>
        <SectionHeading
          id="how"
          eyebrow="Pipeline"
          title="How an image becomes a verdict"
          lead="Every request runs the same deterministic path. Inference is synchronous — about 1.5 seconds — and runs on a worker thread so the API stays responsive."
        />
        <ol className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE.map((step, i) => (
            <li key={step.title} className="clay-sm relative p-5">
              <span className="absolute right-4 top-4 font-display text-xs font-bold text-ink-faint">
                {String(i + 1).padStart(2, '0')}
              </span>
              <span className="clay-sm mb-3 flex h-11 w-11 items-center justify-center text-brand">
                <Icon name={step.icon as IconName} size={20} />
              </span>
              <h3 className="font-display text-[0.95rem] font-semibold text-ink">{step.title}</h3>
              <p className="mt-1.5 text-xs leading-relaxed text-ink-faint">{step.text}</p>
            </li>
          ))}
        </ol>
      </Container>
    </section>
  );
}
