import { useCallback, useEffect, useRef, useState } from 'react'
import { api, type Performance, type Predictions, type Summary, type Verdict } from './api'
import Card from './components/Card'
import { DriftChart, VerdictTable } from './components/DriftChart'
import { HistogramChart, HistogramTable } from './components/HistogramChart'
import { ChallengeTable, DeploymentTable, PerformanceChart, PerformanceTable } from './components/PerformanceChart'
import { PredictionsChart, PredictionsTable, regimeLegend } from './components/PredictionsChart'
import StatTile from './components/StatTile'
import DataTable from './components/DataTable'
import { fmtInt, fmtNum, fmtPct, fmtTime } from './format'

const WINDOWS = [
  { label: '24 h', hours: 24 },
  { label: '7 d', hours: 24 * 7 },
  { label: '30 d', hours: 24 * 30 },
]
const REFRESH_MS = 60_000

interface Data {
  summary: Summary
  predictions: Predictions
  verdicts: Verdict[]
  performance: Performance
}

export default function App() {
  const [hours, setHours] = useState(24)
  const [data, setData] = useState<Data | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [stale, setStale] = useState(false)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const inFlight = useRef(false)

  const load = useCallback(async () => {
    if (inFlight.current) return
    inFlight.current = true
    setStale(true)
    try {
      const [summary, predictions, drift, performance] = await Promise.all([
        api.summary(hours), api.predictions(hours), api.drift(), api.performance(),
      ])
      setData({ summary, predictions, verdicts: drift.verdicts, performance })
      setError(null)
      setUpdatedAt(new Date())
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setStale(false)
      inFlight.current = false
    }
  }, [hours])

  useEffect(() => {
    void load()
    const id = window.setInterval(() => void load(), REFRESH_MS)
    return () => window.clearInterval(id)
  }, [load])

  const s = data?.summary
  const verdict = s?.latest_verdict ?? null
  const verdictState = !verdict ? 'warning' : verdict.drift ? 'critical' : 'good'
  const verdictText = !verdict ? 'no verdict yet' : verdict.drift ? 'DRIFT' : 'no drift'
  const fd002Auc = data?.verdicts.length ? data.verdicts[data.verdicts.length - 1].roc_auc_by_regime.fd002 ?? null : null

  return (
    <main className="app">
      <div className="masthead">
        <div>
          <h1>DriftWatch</h1>
          <div className="sub">Turbofan failure-within-30-cycles prediction, watched for drift. Every prediction is logged; this page reads the log.</div>
        </div>
        <a href="/docs">API docs</a>
      </div>

      <div className="filters" role="toolbar" aria-label="Filters">
        <span>Prediction window</span>
        <div className="group" role="group" aria-label="Window">
          {WINDOWS.map((w) => (
            <button key={w.hours} type="button" aria-pressed={hours === w.hours} onClick={() => setHours(w.hours)}>{w.label}</button>
          ))}
        </div>
        <span className="status">{error ? `error: ${error}` : updatedAt ? `updated ${updatedAt.toLocaleTimeString()}, refreshes every minute` : 'loading…'}</span>
      </div>

      {error && !data ? <div className="error">{error}</div> : null}

      {s ? (
        <div className="tiles">
          <StatTile label="Champion" value={`v${s.champion.version}`} note={`${s.champion.run_name ?? s.champion.model_kind ?? ''}, threshold ${s.champion.threshold.toFixed(4)}`} />
          <StatTile label="Latest drift verdict" value={<span className="status-line"><span className={`status-dot ${verdictState}`} aria-hidden="true" />{verdictText}</span>} note={verdict ? `${fmtTime(verdict.generated_at)}, reference ${verdict.reference_set}` : 'the scheduled detector has not published yet'} />
          <StatTile label="ROC-AUC on replayed regime traffic" value={fd002Auc == null ? '–' : fmtNum(fd002Auc)} note={s.champion.test_roc_auc != null ? `held-out FD001 at registration: ${fmtNum(s.champion.test_roc_auc)}` : undefined} />
          <StatTile label={`Predictions, last ${hours >= 48 ? `${hours / 24} d` : `${hours} h`}`} value={fmtInt(s.predictions.total)} note={`${fmtInt(s.predictions.engines)} engines, ${fmtPct(s.predictions.positive_rate)} flagged`} />
          <StatTile label="Latest deployment" value={s.latest_deployment ? `v${s.latest_deployment.model_version}` : '–'} note={s.latest_deployment ? `${fmtTime(s.latest_deployment.timestamp)}${s.latest_deployment.promotion ? ', human-approved promotion' : ', code push'}` : 'none recorded yet'} />
        </div>
      ) : null}

      <div className="grid">
        <Card
          title="Predictions per hour"
          hint="Stacked by the regime the engine belongs to. FD002 is the quarantined regime replayed as production traffic."
          stale={stale}
          legend={regimeLegend}
          chart={<PredictionsChart buckets={data?.predictions.per_hour ?? []} />}
          table={<PredictionsTable buckets={data?.predictions.per_hour ?? []} />}
        />
        <Card
          title="Failure-probability distribution"
          hint="How confident the champion is across the window. Everything right of the threshold is flagged."
          stale={stale}
          chart={<HistogramChart bins={data?.predictions.histogram ?? []} threshold={s?.champion.threshold ?? 0.5} />}
          table={<HistogramTable bins={data?.predictions.histogram ?? []} threshold={s?.champion.threshold ?? 0.5} />}
        />
        <Card
          title="Drift over time"
          hint="Share of raw input columns that drifted, per regime, each time the detector ran. Each part is compared with the champion's training engines for that regime."
          stale={stale}
          wide
          legend={regimeLegend}
          chart={<DriftChart verdicts={data?.verdicts ?? []} />}
          table={<VerdictTable verdicts={data?.verdicts ?? []} />}
        />
        <Card
          title="Model performance on labeled traffic"
          hint="ROC-AUC of the serving champion on replayed engines whose labels are derivable, per regime. Vertical lines mark human-approved promotions."
          stale={stale}
          wide
          legend={regimeLegend}
          chart={<PerformanceChart labeled={data?.performance.labeled ?? []} deployments={data?.performance.deployments ?? []} />}
          table={<PerformanceTable labeled={data?.performance.labeled ?? []} deployments={data?.performance.deployments ?? []} />}
        />
        <Card
          title="Champion vs challenger"
          hint="Every retrain, judged on the mixed held-out bench (FD001 plus FD002 engines never trained on)."
          stale={stale}
          wide
          chart={<ChallengeTable challenges={data?.performance.challenges ?? []} />}
        />
        <Card
          title="Deployments"
          hint="What served when. Registering a version never changes this list; an approved promotion does."
          stale={stale}
          chart={<DeploymentTable deployments={data?.performance.deployments ?? []} />}
        />
        <Card
          title="Recent predictions"
          hint="The newest records in the log, exactly as the drift monitor reads them."
          stale={stale}
          chart={<RecentTable rows={data?.predictions.recent ?? []} />}
        />
      </div>
    </main>
  )
}

function RecentTable({ rows }: { rows: Predictions['recent'] }) {
  return (
    <DataTable
      rows={rows}
      rowKey={(r, i) => `${r.timestamp}-${r.unit}-${r.cycle}-${i}`}
      empty="No predictions in this window"
      columns={[
        { key: 'when', label: 'Time', render: (r) => fmtTime(r.timestamp) },
        { key: 'engine', label: 'Engine', render: (r) => `${r.regime.toUpperCase()} unit ${r.unit}` },
        { key: 'cycle', label: 'Cycle', num: true, render: (r) => fmtInt(r.cycle) },
        { key: 'p', label: 'Probability', num: true, render: (r) => fmtNum(r.probability) },
        { key: 'label', label: 'Flagged', render: (r) => (r.label ? 'yes' : 'no') },
        { key: 'version', label: 'Model', render: (r) => `v${r.model_version}` },
        { key: 'latency', label: 'Server ms', num: true, render: (r) => fmtNum(r.latency_ms, 1) },
      ]}
    />
  )
}
