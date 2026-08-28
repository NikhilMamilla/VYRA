import { Container } from '../ui/primitives';
import { Wordmark } from './Wordmark';

const STACK = ['React', 'TypeScript', 'FastAPI', 'scikit-learn', 'OpenCV', 'PostgreSQL', 'Docker'];

export function Footer() {
  return (
    <footer className="border-t border-line/60 py-12">
      <Container className="flex flex-col items-start gap-6 sm:flex-row sm:items-center sm:justify-between">
        <div className="space-y-2">
          <Wordmark />
          <p className="max-w-sm text-xs leading-relaxed text-ink-faint">
            AI-powered image quality &amp; defect detection. Built for a technical assessment —
            classical computer vision, a calibrated learned model, and an honest account of what it
            can and cannot do.
          </p>
        </div>
        <ul className="flex flex-wrap gap-1.5">
          {STACK.map((t) => (
            <li
              key={t}
              className="rounded-full bg-ink/[0.04] px-2.5 py-1 text-[0.7rem] font-medium text-ink-soft"
            >
              {t}
            </li>
          ))}
        </ul>
      </Container>
    </footer>
  );
}
