import { useEffect, useState } from 'react';

import { ThemeToggle } from '../../theme/ThemeToggle';
import type { HealthState } from '../../hooks/useBackendHealth';
import { Icon } from '../ui/Icon';
import { Wordmark } from './Wordmark';

const LINKS = [
  ['#analyze', 'Analyze'],
  ['#how', 'How it works'],
  ['#model', 'Under the hood'],
  ['#metrics', 'Metrics'],
];

export function NavBar({ health }: { health: HealthState }) {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  const analyzer =
    health.status === 'success' ? health.health.components.analyzer : null;
  const modelOk = analyzer?.status === 'ok';

  return (
    <header
      className={`sticky top-0 z-40 transition-all duration-300 ${
        scrolled ? 'glass border-b border-line/60' : 'border-b border-transparent'
      }`}
    >
      <div className="mx-auto flex h-16 w-full max-w-6xl items-center gap-4 px-5 sm:px-6">
        <a href="#top" className="shrink-0">
          <Wordmark />
        </a>

        <nav className="ml-4 hidden items-center gap-1 md:flex">
          {LINKS.map(([href, label]) => (
            <a
              key={href}
              href={href}
              className="rounded-lg px-3 py-1.5 text-sm font-medium text-ink-soft transition-colors hover:bg-ink/[0.04] hover:text-ink"
            >
              {label}
            </a>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-3">
          <span
            className="hidden items-center gap-1.5 rounded-full bg-ink/[0.04] px-2.5 py-1 text-xs font-medium sm:flex"
            title={analyzer?.detail ?? undefined}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                health.status === 'error'
                  ? 'bg-poor'
                  : modelOk
                    ? 'bg-good'
                    : 'bg-degraded'
              }`}
            />
            <span className="text-ink-soft">
              {health.status === 'loading'
                ? 'connecting'
                : health.status === 'error'
                  ? 'API offline'
                  : modelOk
                    ? health.health.analyzer_model_version
                    : 'model not loaded'}
            </span>
          </span>
          <ThemeToggle />
          <a
            href="#analyze"
            className="hidden items-center gap-1.5 rounded-xl bg-brand px-3.5 py-2 text-sm font-semibold text-white shadow-glow transition hover:brightness-110 sm:inline-flex"
          >
            Analyze <Icon name="arrowRight" size={14} />
          </a>
        </div>
      </div>
    </header>
  );
}
