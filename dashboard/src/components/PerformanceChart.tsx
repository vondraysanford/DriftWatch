import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { Challenge, Deployment, LabeledPoint } from '../api'
import { fmtInt, fmtNum, fmtTime, tickFormatter } from '../format'
import ChartTip from './ChartTip'
import DataTable from './DataTable'

interface Props {
  labeled: LabeledPoint[]
  deployments: Deployment[]
}

interface Row {
  t: number
  when: string
  fd001: number | null
  fd002: number | null
  records: Record<string, number>
}

const axisTick = { fill: 'var(--muted)', fontSize: 12 }
const dot = (color: string) => ({ r: 4, strokeWidth: 2, stroke: 'var(--surface)', fill: color })

function toRows(points: LabeledPoint[]): Row[] {
  const byTime = new Map<string, Row>()
  for (const p of points) {
    const row = byTime.get(p.generated_at) ?? { t: new Date(p.generated_at).getTime(), when: p.generated_at, fd001: null, fd002: null, records: {} }
    row[p.regime] = p.roc_auc
    row.records[p.regime] = p.records
    byTime.set(p.generated_at, row)
  }
  return [...byTime.values()].sort((a, b) => a.t - b.t)
}

// The champion's ROC-AUC on labeled production traffic, per regime, over the detector's runs.
// Promotions are the vertical reference lines: the model behind the line changes there.
export function PerformanceChart({ labeled, deployments }: Props) {
  const rows = toRows(labeled)
  if (rows.length === 0) return <div className="empty">No labeled traffic scored yet</div>
  const promotions = deployments.filter((d) => d.promotion)
  const span = rows[rows.length - 1].t - rows[0].t
  return (
    <ResponsiveContainer width="100%" height={260}>
      {/* Straight segments between runs: each point is a discrete detector run, not a continuous measurement. */}
      <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="t" type="number" domain={['dataMin', 'dataMax']} tickFormatter={tickFormatter(span)} tick={axisTick} axisLine={{ stroke: 'var(--axis)' }} tickLine={false} minTickGap={40} />
        <YAxis domain={[0.4, 1]} tickFormatter={(v: number) => v.toFixed(2)} tick={axisTick} axisLine={false} tickLine={false} />
        {promotions.map((d) => (
          <ReferenceLine key={d.timestamp} x={new Date(d.timestamp).getTime()} stroke="var(--ink-2)" strokeDasharray="4 3" label={{ value: `v${d.promotion} promoted`, position: 'insideTopRight', fill: 'var(--ink-2)', fontSize: 12 }} />
        ))}
        <Tooltip
          cursor={{ stroke: 'var(--axis)' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const row = payload[0].payload as Row
            return (
              <ChartTip
                title={fmtTime(row.when)}
                rows={(['fd001', 'fd002'] as const)
                  .filter((r) => row[r] != null)
                  .map((r) => ({ label: `${r.toUpperCase()} (${fmtInt(row.records[r])} labeled)`, value: fmtNum(row[r]), color: r === 'fd002' ? 'var(--series-2)' : 'var(--series-1)' }))}
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

export function PerformanceTable({ labeled }: Props) {
  return (
    <DataTable
      rows={[...labeled].reverse()}
      rowKey={(p, i) => `${p.generated_at}-${p.regime}-${i}`}
      empty="No labeled traffic scored yet"
      columns={[
        { key: 'when', label: 'Run', render: (p) => fmtTime(p.generated_at) },
        { key: 'regime', label: 'Traffic', render: (p) => p.regime.toUpperCase() },
        { key: 'reference', label: 'Reference', render: (p) => p.reference_set },
        { key: 'auc', label: 'ROC-AUC', num: true, render: (p) => fmtNum(p.roc_auc) },
        { key: 'precision', label: 'Precision', num: true, render: (p) => fmtNum(p.precision, 3) },
        { key: 'recall', label: 'Recall', num: true, render: (p) => fmtNum(p.recall, 3) },
        { key: 'records', label: 'Labeled', num: true, render: (p) => fmtInt(p.records) },
      ]}
    />
  )
}

export function ChallengeTable({ challenges }: { challenges: Challenge[] }) {
  return (
    <DataTable
      rows={[...challenges].reverse()}
      rowKey={(c) => c.published_at}
      empty="No champion-vs-challenger results published yet"
      columns={[
        { key: 'when', label: 'Retrain', render: (c) => fmtTime(c.published_at) },
        { key: 'champion', label: 'Champion', render: (c) => `v${c.champion_version}` },
        { key: 'champion_auc', label: 'Champion bench ROC-AUC', num: true, render: (c) => `${fmtNum(c.champion_roc_auc)} (FD001 ${fmtNum(c.champion_by_regime.fd001)}, FD002 ${fmtNum(c.champion_by_regime.fd002)})` },
        { key: 'challenger', label: 'Challenger', render: (c) => c.challenger_run },
        { key: 'challenger_auc', label: 'Challenger bench ROC-AUC', num: true, render: (c) => `${fmtNum(c.challenger_roc_auc)} (FD001 ${fmtNum(c.challenger_by_regime.fd001)}, FD002 ${fmtNum(c.challenger_by_regime.fd002)})` },
        { key: 'gain', label: 'Gain', num: true, render: (c) => `${c.gain >= 0 ? '+' : ''}${fmtNum(c.gain)} (margin ${c.margin})` },
        { key: 'result', label: 'Result', render: (c) => (c.registered_version ? `registered v${c.registered_version}` : 'champion holds') },
      ]}
    />
  )
}

export function DeploymentTable({ deployments }: { deployments: Deployment[] }) {
  return (
    <DataTable
      rows={[...deployments].reverse()}
      rowKey={(d) => d.timestamp}
      empty="No deployments recorded yet"
      columns={[
        { key: 'when', label: 'Deployed', render: (d) => fmtTime(d.timestamp) },
        { key: 'version', label: 'Model version', render: (d) => `v${d.model_version}` },
        { key: 'promotion', label: 'Promotion', render: (d) => (d.promotion ? `v${d.promotion}, human-approved` : 'none (code push)') },
        { key: 'image', label: 'Image tag', render: (d) => <span className="mono">{d.image_tag}</span> },
        { key: 'run', label: 'Run', render: (d) => (d.run_url ? <a href={d.run_url}>workflow run</a> : '–') },
      ]}
    />
  )
}
