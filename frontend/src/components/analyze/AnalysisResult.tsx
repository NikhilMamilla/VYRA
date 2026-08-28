import type { ReactNode } from 'react';

import type { Analysis } from '../../lib/api/types';
import { formatBytes, formatDateTime } from '../../lib/format';
import { ExplanationPanel } from './ExplanationPanel';
import { ImageCanvas } from './ImageCanvas';
import { IssueList } from './IssueList';
import { QualityScore } from './QualityScore';
import { StatisticsGrid } from './StatisticsGrid';

interface Props {
  analysis: Analysis;
  previewUrl?: string | null;
}

function Block({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div>
      <h4 className="mb-3 text-[0.7rem] font-semibold uppercase tracking-[0.16em] text-ink-faint">
        {title}
      </h4>
      {children}
    </div>
  );
}

export function AnalysisResult({ analysis, previewUrl }: Props) {
  return (
    <div className="animate-fade-up space-y-7">
      <div className="grid gap-6 md:grid-cols-[minmax(0,1fr)_auto] md:items-center">
        {previewUrl ? (
          <ImageCanvas
            src={previewUrl}
            alt={analysis.image.filename}
            defect={analysis.explanation.potential_defect}
          />
        ) : (
          <div className="clay-sm grid place-items-center px-4 py-14 text-center text-sm text-ink-faint">
            Image preview is shown only for the current upload.
          </div>
        )}
        {analysis.quality_score != null && analysis.quality_label != null && (
          <QualityScore score={analysis.quality_score} label={analysis.quality_label} />
        )}
      </div>

      <dl className="flex flex-wrap gap-x-5 gap-y-1 text-xs text-ink-faint">
        {[
          ['file', analysis.image.filename],
          analysis.image.width
            ? ['dimensions', `${analysis.image.width}×${analysis.image.height}`]
            : null,
          ['size', formatBytes(analysis.image.size_bytes)],
          ['model', analysis.model_version],
          ['analyzed', formatDateTime(analysis.created_at)],
        ]
          .filter((x): x is [string, string] => x !== null)
          .map(([k, v]) => (
            <div key={k}>
              <dt className="inline">{k}: </dt>
              <dd className="inline text-ink-soft">{v}</dd>
            </div>
          ))}
      </dl>

      <div className="rule" />

      <Block title="Detected issues">
        <IssueList issues={analysis.issues} />
      </Block>

      <Block title="Image statistics">
        <StatisticsGrid metrics={analysis.metrics} />
      </Block>

      <Block title="Explanation">
        <ExplanationPanel explanation={analysis.explanation} />
      </Block>
    </div>
  );
}
