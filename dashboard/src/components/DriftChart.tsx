import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Verdict } from '../api'
import { fmtInt, fmtPct, fmtPct0, fmtTime, tickFormatter } from '../format'
import ChartTip from './ChartTip'
import DataTable from './DataTable'

interface Props {
  verdicts: Verdict[]
  driftShare?: number
}

interface Row {
  t: number
  verdict: Verdict
  fd001: number | null
  fd002: number | null
}

const axisTick = { fill: 'var(--muted)', fontSize: 12 }
const dot = (color: string) => ({ r: 4, strokeWidth: 2, stroke: 'var(--surface)', fill: color })

function toRows(verdicts: Verdict[]): Row[] {
  return verdicts
    .filter((v) => v.generated_at)
    .map((v) => {
      const part = (regime: string) => v.parts.find((p) => p.regime === regime)
      const share = (regime: string) => {
        const p = part(regime)
        return p && !p.skipped ? p.raw_share : null
      }
      return { t: new Date(v.generated_at).getTime(), verdict: v, fd001: share('fd001'), fd002: share('fd002') }
    })
}

// Share of drifted raw-input columns per verdict, one line per regime part, against the
// threshold that declares drift. Each point is a scheduled run of the detector.
export function DriftChart({ verdicts, driftShare = 0.3 }: Props) {
  const rows = toRows(verdicts)
  if (rows.length === 0) return <div className="empty">No drift verdicts published yet</div>
  const span = rows[rows.length - 1].t - rows[0].t
  return (
    <ResponsiveContainer width="100%" height={260}>
      {/* Straight segments between runs: each point is a discrete detector run, not a continuous measurement. */}
      <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tickFormatter={tickFormatter(span)} tick={axisTick} axisLine={{ stroke: 'var(--axis)' }} tickLine={false} minTickGap={40} />
        <YAxis domain={[0, 1]} tickFormatter={fmtPct0} tick={axisTick} axisLine={false} tickLine={false} />
        <ReferenceLine y={driftShare} stroke="var(--ink-2)" strokeDasharray="4 3" label={{ value: `drift at ${fmtPct(driftShare)} of raw columns`, position: 'insideTopLeft', fill: 'var(--ink-2)', fontSize: 12 }} />
        <Tooltip
          cursor={{ stroke: 'var(--axis)' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const row = payload[0].payload as Row
            const v = row.verdict
            return (
              <ChartTip
                title={`${fmtTime(v.generated_at)}: ${v.drift ? 'DRIFT' : v.insufficient_data ? 'no verdict' : 'no drift'} (reference ${v.reference_set})`}
                rows={v.parts.map((p) => ({
                  label: `${p.regime.toUpperCase()} vs ${p.compared_to ?? '–'}`,
                  value: p.skipped ? `skipped (${fmtInt(p.records)} records)` : `${fmtInt(p.raw_count)} of ${fmtInt(p.raw_columns)} raw, ${fmtPct(p.features_share)} features`,
                  color: p.regime === 'fd002' ? 'var(--series-2)' : 'var(--series-1)',
                }))}
              />
            )
          }}
        />
        <Line type="linear" dataKey="fd001" stroke="var(--series-1)" strokeWidth={2} dot={dot('var(--series-1)')} activeDot={{ r: 6 }} connectNulls isAnimationActive={false} />
        <Line type="linear" dataKey="fd002" stroke="var(--series-2)" strokeWidth={2} dot={dot('var(--series-2)')} activeDot={{ r: 6 }} connectNulls isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

export function VerdictTable({ verdicts }: Props) {
  return (
    <DataTable
      rows={[...verdicts].reverse()}
      rowKey={(v) => v.published_at}
      empty="No drift verdicts published yet"
      columns={[
        { key: 'when', label: 'Run', render: (v) => fmtTime(v.generated_at) },
        {
          key: 'verdict',
          label: 'Verdict',
          render: (v) => (
            <span className="pill">
              <span className={`status-dot ${v.drift ? 'critical' : v.insufficient_data ? 'warning' : 'good'}`} aria-hidden="true" />
              {v.drift ? 'DRIFT' : v.insufficient_data ? 'no verdict' : 'no drift'}
            </span>
          ),
        },
        { key: 'reference', label: 'Reference', render: (v) => v.reference_set },
        { key: 'records', label: 'Records', num: true, render: (v) => fmtInt(v.records) },
        {
          key: 'parts',
          label: 'Per regime (raw columns drifted)',
          render: (v) => v.parts.map((p) => (p.skipped ? `${p.regime}: skipped` : `${p.regime} vs ${p.compared_to}: ${p.raw_count} of ${p.raw_columns}${p.settings_drifted.length ? ` (settings: ${p.settings_drifted.join(', ')})` : ''}`)).join(' · '),
        },
        { key: 'auc', label: 'Champion ROC-AUC on labeled traffic', num: true, render: (v) => (v.roc_auc == null ? '–' : v.roc_auc.toFixed(4)) },
      ]}
    />
  )
}
