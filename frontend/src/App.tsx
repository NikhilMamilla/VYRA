import { Workspace } from './components/analyze/Workspace';
import { Footer } from './components/layout/Footer';
import { Hero } from './components/layout/Hero';
import { NavBar } from './components/layout/NavBar';
import { HonestMetrics } from './components/marketing/HonestMetrics';
import { HowItWorks } from './components/marketing/HowItWorks';
import { UnderTheHood } from './components/marketing/UnderTheHood';
import { useBackendHealth } from './hooks/useBackendHealth';

export default function App() {
  const health = useBackendHealth();

  const analyzerReady =
    health.status === 'loading'
      ? null
      : health.status === 'success' && health.health.components.analyzer?.status === 'ok';

  return (
    <div className="relative isolate min-h-screen">
      <NavBar health={health} />
      <main>
        <Hero />
        <Workspace analyzerReady={analyzerReady} />
        <HowItWorks />
        <UnderTheHood />
        <HonestMetrics />
      </main>
      <Footer />
    </div>
  );
}
