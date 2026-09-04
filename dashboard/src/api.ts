// Typed access to the metrics API served by the same container (serving/metrics.py).

export type Regime = 'fd001' | 'fd002'

export interface Summary {
  champion: {
    name: string
    version: string
    run_name: string | null
    model_kind: string | null
    threshold: number
    test_roc_auc: number | null
    data_version: string | null
  }
  window_hours: number
  predictions: {
    total: number
    engines: number
    by_regime: Record<string, number>
    by_model_version: Record<string, number>
    positive_rate: number | null
    first: string | null
    last: string | null
  }
  latest_verdict: { generated_at: string; drift: boolean; reason: string; reference_set: string; records: number } | null
  latest_challenge: {
    published_at: string
    champion_version: string
    champion_roc_auc: number
    challenger_roc_auc: number
    registered_version: string | null
  } | null
  latest_deployment: { timestamp: string; model_version: string; image_tag: string; promotion: string | null; run_url: string | null } | null
  sink: string
}

export interface HourBucket {
  hour: string
  fd001?: number
  fd002?: number
  total: number
}

export interface HistogramBin {
  bin: string
  low: number
  count: number
}

export interface RecentPrediction {
  timestamp: string
  unit: number
  cycle: number
  regime: Regime
  probability: number
  label: number
  model_version: string
  latency_ms: number
}

export interface Predictions {
  window_hours: number
  per_hour: HourBucket[]
  histogram: HistogramBin[]
  recent: RecentPrediction[]
}

export interface VerdictPart {
  regime: Regime
  compared_to: string | null
  records: number
  raw_share: number | null
  raw_count: number | null
  raw_columns: number | null
  features_share: number | null
  settings_drifted: string[]
  drift: boolean
  skipped: string | null
}

export interface Verdict {
  generated_at: string
  published_at: string
  drift: boolean
  insufficient_data: boolean
  reference_set: string
  records: number
  window_hours: number
  reason: string
  parts: VerdictPart[]
  roc_auc: number | null
  roc_auc_reference: number | null
  roc_auc_by_regime: Record<string, number | null>
}

export interface LabeledPoint {
  generated_at: string
  regime: Regime
  roc_auc: number
  precision: number | null
  recall: number | null
  records: number
  reference_set: string
}

export interface Challenge {
  published_at: string
  champion_version: string
  champion_roc_auc: number
  champion_by_regime: Record<string, number>
  challenger_run: string
  challenger_roc_auc: number
  challenger_by_regime: Record<string, number>
  gain: number
  margin: number
  registered_version: string | null
}

export interface Deployment {
  timestamp: string
  model_version: string
  image_tag: string
  promotion: string | null
  run_url: string | null
}

export interface Performance {
  labeled: LabeledPoint[]
  challenges: Challenge[]
  deployments: Deployment[]
}

// Empty when the dashboard is served by the API's own container (same origin, /dashboard);
// the Container App's URL when it is hosted on Cloudflare Pages, like the sibling projects.
export const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')

export async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { headers: { Accept: 'application/json' } })
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`)
  return response.json() as Promise<T>
}

export const api = {
  summary: (hours: number) => getJSON<Summary>(`/api/summary?hours=${hours}`),
  predictions: (hours: number) => getJSON<Predictions>(`/api/predictions?hours=${hours}&recent=25`),
  drift: () => getJSON<{ verdicts: Verdict[] }>('/api/drift'),
  performance: () => getJSON<Performance>('/api/performance'),
}
