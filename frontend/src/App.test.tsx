import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import { ApiError } from './lib/api/client';
import * as endpoints from './lib/api/endpoints';
import { ThemeProvider } from './theme/ThemeProvider';
import type { Analysis, Health } from './lib/api/types';

vi.mock('./lib/api/endpoints');

const health: Health = {
  status: 'ok',
  version: '0.3.0',
  environment: 'test',
  uptime_seconds: 12,
  analyzer_model_version: 'vyra-quality-model-v1',
  components: {
    database: { status: 'ok', detail: null, latency_ms: 1 },
    storage: { status: 'ok', detail: null, latency_ms: null },
    analyzer: { status: 'ok', detail: 'model vyra-quality-model-v1', latency_ms: null },
  },
};

const analysis: Analysis = {
  id: 'abc-123',
  created_at: new Date().toISOString(),
  status: 'completed',
  image: {
    filename: 'photo.jpg',
    content_type: 'image/jpeg',
    size_bytes: 2048,
    width: 640,
    height: 480,
  },
  quality_score: 62,
  quality_label: 'DEGRADED',
  model_version: 'vyra-quality-model-v1',
  issues: [
    { type: 'blur', severity: 'medium', confidence: 0.71, validation: 'real-world', detail: null },
  ],
  metrics: { sharpness: 12.3, brightness: 0.42 },
  explanation: { summary: 'Quality score 62/100 (DEGRADED). Flagged: blur.', evidence: [] },
  error_message: null,
};

function renderApp() {
  return render(
    <ThemeProvider>
      <App />
    </ThemeProvider>,
  );
}

beforeEach(() => {
  localStorage.clear();
  vi.mocked(endpoints.getHealth).mockResolvedValue(health);
  vi.mocked(endpoints.listAnalyses).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 });
  vi.mocked(endpoints.createAnalysis).mockResolvedValue(analysis);
});

afterEach(() => vi.clearAllMocks());

describe('App', () => {
  it('shows a loading then a ready state with the model version', async () => {
    renderApp();
    expect(screen.getByText(/connecting/i)).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getAllByText(/vyra-quality-model-v1/i).length).toBeGreaterThan(0),
    );
    expect(screen.getByText(/Drop an image or click to browse/i)).toBeInTheDocument();
  });

  it('renders the analysis result after uploading (result state)', async () => {
    renderApp();
    await screen.findAllByText(/vyra-quality-model-v1/i);

    const file = new File(['x'], 'photo.jpg', { type: 'image/jpeg' });
    await userEvent.upload(document.querySelector('input[type=file]')!, file);
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }));

    const workspace = document.querySelector('#analyze')!;
    expect(await within(workspace as HTMLElement).findByText('62')).toBeInTheDocument();
    const ws = within(workspace as HTMLElement);
    expect(ws.getByText('DEGRADED')).toBeInTheDocument();
    expect(ws.getByText('Blur')).toBeInTheDocument();
    expect(ws.getByText('71%')).toBeInTheDocument();
    expect(ws.getByText(/Detected issues/i)).toBeInTheDocument();
  });

  it('shows a friendly message when analysis fails (error state)', async () => {
    vi.mocked(endpoints.createAnalysis).mockRejectedValue(new ApiError(422, 'invalid_image', 'bad'));
    renderApp();
    await screen.findAllByText(/vyra-quality-model-v1/i);

    const file = new File(['x'], 'photo.jpg', { type: 'image/jpeg' });
    await userEvent.upload(document.querySelector('input[type=file]')!, file);
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }));

    expect(await screen.findByText(/could not be read as an image/i)).toBeInTheDocument();
  });

  it('lists prior analyses in the history panel (history state)', async () => {
    vi.mocked(endpoints.listAnalyses).mockResolvedValue({
      items: [analysis],
      total: 1,
      limit: 20,
      offset: 0,
    });
    renderApp();
    const historyHeading = await screen.findByRole('heading', { name: /history/i });
    const panel = historyHeading.closest('section')!;
    expect(await within(panel).findByText('photo.jpg')).toBeInTheDocument();
  });

  it('degrades when the API is unreachable', async () => {
    vi.mocked(endpoints.getHealth).mockRejectedValue(new Error('boom'));
    renderApp();
    expect(await screen.findByText(/API offline/i)).toBeInTheDocument();
  });

  it('toggles the theme and persists the choice', async () => {
    renderApp();
    await screen.findAllByText(/vyra-quality-model-v1/i);
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    await userEvent.click(screen.getByRole('switch', { name: /dark theme/i }));
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
    expect(localStorage.getItem('vyra-theme')).toBe('dark');
  });
});
