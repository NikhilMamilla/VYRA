import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import { ApiError } from './lib/api/client';
import * as endpoints from './lib/api/endpoints';
import type { Analysis, Health } from './lib/api/types';

vi.mock('./lib/api/endpoints');

const health: Health = {
  status: 'ok',
  version: '0.3.0',
  environment: 'test',
  uptime_seconds: 12,
  analyzer_model_version: 'vyra-quality-model-v1',
  components: { analyzer: { status: 'ok', detail: 'model vyra-quality-model-v1', latency_ms: null } },
};

const analysis: Analysis = {
  id: 'abc-123',
  created_at: '2026-08-28T10:00:00Z',
  status: 'completed',
  image: { filename: 'photo.jpg', content_type: 'image/jpeg', size_bytes: 2048, width: 640, height: 480 },
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

beforeEach(() => {
  vi.mocked(endpoints.getHealth).mockResolvedValue(health);
  vi.mocked(endpoints.listAnalyses).mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 });
  vi.mocked(endpoints.createAnalysis).mockResolvedValue(analysis);
});

afterEach(() => vi.clearAllMocks());

describe('App', () => {
  it('shows a loading state then the model badge (success state)', async () => {
    render(<App />);
    expect(screen.getByText(/checking server/i)).toBeInTheDocument();
    await waitFor(() =>
      expect(screen.getByText(/model vyra-quality-model-v1/i)).toBeInTheDocument(),
    );
    expect(screen.getByText(/Drop an image or click to browse/i)).toBeInTheDocument();
  });

  it('renders the analysis result after uploading (result state)', async () => {
    render(<App />);
    await screen.findByText(/model vyra-quality-model-v1/i);

    const file = new File(['x'], 'photo.jpg', { type: 'image/jpeg' });
    await userEvent.upload(document.querySelector('input[type=file]')!, file);
    await userEvent.click(screen.getByRole('button', { name: /analyze image/i }));

    expect(await screen.findByText('62')).toBeInTheDocument();
    expect(screen.getByText('DEGRADED')).toBeInTheDocument();
    expect(screen.getByText('Blur')).toBeInTheDocument();
    expect(screen.getByText(/71% confidence/i)).toBeInTheDocument();
  });

  it('shows a friendly message when analysis fails (error state)', async () => {
    vi.mocked(endpoints.createAnalysis).mockRejectedValue(
      new ApiError(422, 'invalid_image', 'bad'),
    );
    render(<App />);
    await screen.findByText(/model vyra-quality-model-v1/i);

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
    render(<App />);
    expect(await screen.findByText('photo.jpg')).toBeInTheDocument();
    expect(screen.getByText('1 total')).toBeInTheDocument();
  });

  it('degrades when the API is unreachable', async () => {
    vi.mocked(endpoints.getHealth).mockRejectedValue(new Error('boom'));
    render(<App />);
    expect(await screen.findByText(/API unreachable/i)).toBeInTheDocument();
  });
});
