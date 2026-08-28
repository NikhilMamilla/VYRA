import { MODEL, TIERS } from '../../content/model';
import { Icon } from '../ui/Icon';
import { Badge, Card, Container, SectionHeading } from '../ui/primitives';

const SPEC = [
  ['Model version', MODEL.version],
  ['Feature version', MODEL.featureVersion],
  ['Family', MODEL.family],
  ['Training data', MODEL.training],
  ['Calibration', MODEL.calibration],
];

export function UnderTheHood() {
  return (
    <section className="py-16 sm:py-24">
      <Container>
        <SectionHeading
          id="model"
          eyebrow="Under the hood"
          title="One self-describing model bundle"
          lead="No CNN — a calibrated RandomForest over 42 interpretable features. The three real-world issues are trained on real VizWiz photos; the bundle pins every threshold, the calibration and the training run, so the backend hard-codes nothing."
        />

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
          <Card surface="glass" className="p-6">
            <h3 className="mb-4 flex items-center gap-2 font-display text-sm font-semibold text-ink">
              <Icon name="cpu" size={16} className="text-brand" /> Model card
            </h3>
            <dl className="space-y-3">
              {SPEC.map(([k, v]) => (
                <div key={k} className="flex flex-col gap-0.5 border-b border-line/60 pb-3 last:border-0 last:pb-0">
                  <dt className="text-[0.7rem] uppercase tracking-wide text-ink-faint">{k}</dt>
                  <dd className="text-sm text-ink">{v}</dd>
                </div>
              ))}
            </dl>
          </Card>

          <div className="grid gap-4">
            {TIERS.map((tier) => (
              <Card key={tier.key} surface="clay-sm" className="p-5">
                <div className="mb-2 flex flex-wrap items-center gap-2">
                  <h3 className="font-display text-[0.95rem] font-semibold text-ink">{tier.title}</h3>
                  {tier.issues.map((iss) => (
                    <Badge key={iss} tone={tier.tone}>
                      {iss}
                    </Badge>
                  ))}
                </div>
                <p className="text-xs leading-relaxed text-ink-soft">{tier.body}</p>
              </Card>
            ))}
          </div>
        </div>
      </Container>
    </section>
  );
}
