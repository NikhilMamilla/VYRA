import { DISCLAIMERS, METRICS } from '../../content/model';
import { ISSUE_LABELS } from '../../lib/format';
import { Icon } from '../ui/Icon';
import { Card, Container, SectionHeading } from '../ui/primitives';

export function HonestMetrics() {
  return (
    <section className="bg-bg-alt/60 py-16 sm:py-24">
      <Container>
        <SectionHeading
          id="metrics"
          eyebrow="Honest metrics"
          title="Synthetic performance is not real-world performance"
          lead="blur, underexposure and overexposure are now trained on real VizWiz photos and evaluated once on a held-out real sample — that lifted real macro-F1 from 0.43 to 0.54. noise and corruption are still synthetic-only. Every number, and which tier each issue is in, is stated in the API response, on every issue badge, and here."
        />

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
          <Card surface="glass" className="overflow-hidden">
            <div className="grid grid-cols-[1fr_auto_auto] gap-x-4 border-b border-line/60 px-5 py-3 text-[0.7rem] font-semibold uppercase tracking-wide text-ink-faint">
              <span>Issue</span>
              <span className="text-right">Synthetic F1</span>
              <span className="text-right">Real-world F1</span>
            </div>
            {METRICS.synthetic.map(([issue, syn], i) => (
              <div
                key={issue}
                className="grid grid-cols-[1fr_auto_auto] items-center gap-x-4 border-b border-line/40 px-5 py-2.5 text-sm last:border-0"
              >
                <span className="text-ink">{ISSUE_LABELS[issue] ?? issue}</span>
                <span className="text-right tabular-nums text-ink-soft">{syn}</span>
                <span
                  className={`text-right font-semibold tabular-nums ${
                    METRICS.real[i][1] === '—' ? 'text-ink-faint' : 'text-ink'
                  }`}
                >
                  {METRICS.real[i][1]}
                </span>
              </div>
            ))}
            <div className="grid grid-cols-[1fr_auto_auto] gap-x-4 bg-brand/[0.05] px-5 py-3 text-sm font-semibold">
              <span className="text-ink">macro-F1</span>
              <span className="text-right tabular-nums text-ink-soft">{METRICS.headline.synthetic}</span>
              <span className="text-right tabular-nums text-brand">
                {METRICS.headline.real}
                <span className="ml-1.5 font-normal text-ink-faint">(was {METRICS.previousReal})</span>
              </span>
            </div>
            <p className="px-5 py-3 text-xs text-ink-faint">
              “—” = no real-world evaluation exists (VizWiz-QualityIssues has no noise or
              corruption labels). Real-world macro-F1 is over blur / underexposure / overexposure,
              read once on a held-out sample. “was 0.43” = the earlier synthetic-trained model.
            </p>
          </Card>

          <div className="grid gap-4">
            {DISCLAIMERS.map((d) => (
              <Card key={d.title} surface="clay-sm" className="p-5">
                <h3 className="mb-1.5 flex items-center gap-2 font-display text-[0.9rem] font-semibold text-ink">
                  <Icon name="alert" size={14} className="text-degraded" />
                  {d.title}
                </h3>
                <p className="text-xs leading-relaxed text-ink-soft">{d.body}</p>
              </Card>
            ))}
          </div>
        </div>
      </Container>
    </section>
  );
}
