import type { Analysis } from '../lib/api/types';
import { formatBytes, formatDateTime } from '../lib/format';
import { ExplanationPanel } from './ExplanationPanel';
import { ImageCanvas } from './ImageCanvas';
import { IssueList } from './IssueList';
import { QualityScore } from './QualityScore';
import { StatisticsGrid } from './StatisticsGrid';

interface Props {
  analysis: Analysis;
  /** Local object URL when the image was just uploaded; history items have none. */
  previewUrl?: string | null;
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-400">{title}</h3>
      {children}
    </div>
  );
}

export function AnalysisResult({ analysis, previewUrl }: Props) {
  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
        {previewUrl ? (
          <ImageCanvas
            src={previewUrl}
            alt={analysis.image.filename}
            defect={analysis.explanation.potential_defect}
          />
        ) : (
          <div className="rounded-xl border border-dashed border-slate-200 px-4 py-10 text-center text-sm text-slate-400">
            Image preview is only shown for the current upload.
          </div>
        )}
        {analysis.quality_score != null && analysis.quality_label != null && (
          <QualityScore score={analysis.quality_score} label={analysis.quality_label} />
        )}
      </div>

      <dl className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-400">
        <div>
          <dt className="inline">file: </dt>
          <dd className="inline text-slate-500">{analysis.image.filename}</dd>
        </div>
        {analysis.image.width && (
          <div>
            <dt className="inline">dimensions: </dt>
            <dd className="inline text-slate-500">
              {analysis.image.width}×{analysis.image.height}
            </dd>
          </div>
        )}
        <div>
          <dt className="inline">size: </dt>
          <dd className="inline text-slate-500">{formatBytes(analysis.image.size_bytes)}</dd>
        </div>
        <div>
          <dt className="inline">model: </dt>
          <dd className="inline text-slate-500">{analysis.model_version}</dd>
        </div>
        <div>
          <dt className="inline">analyzed: </dt>
          <dd className="inline text-slate-500">{formatDateTime(analysis.created_at)}</dd>
        </div>
      </dl>

      <Section title="Detected issues">
        <IssueList issues={analysis.issues} />
      </Section>

      <Section title="Image statistics">
        <StatisticsGrid metrics={analysis.metrics} />
      </Section>

      <Section title="Explanation">
        <ExplanationPanel explanation={analysis.explanation} />
      </Section>
    </div>
  );
}
