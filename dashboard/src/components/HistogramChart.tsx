import { Bar, BarChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { HistogramBin } from '../api'
import { fmtInt } from '../format'
import ChartTip from './ChartTip'
import DataTable from './DataTable'

interface Props {
  bins: HistogramBin[]
  threshold: number
}

const axisTick = { fill: 'var(--muted)', fontSize: 12 }

// One series, one hue (slot 1), no legend: the title says what is plotted. The operating
// threshold is the one line that means something, so it is the one dashed reference line.
export function HistogramChart({ bins, threshold }: Props) {
  if (bins.length === 0) return <div className="empty">No predictions in this window</div>
  const thresholdBin = bins.findIndex((b) => threshold >= b.low && threshold < b.low + 0.1)
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={bins} margin={{ top: 16, right: 8, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="bin" tick={axisTick} axisLine={{ stroke: 'var(--axis)' }} tickLine={false} interval={0} />
        <YAxis tick={axisTick} axisLine={false} tickLine={false} allowDecimals={false} tickFormatter={(v: number) => fmtInt(v)} />
        <Tooltip
          cursor={{ fill: 'var(--series-1-wash)' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const row = payload[0].payload as HistogramBin
            return <ChartTip title={`Probability ${row.bin}`} rows={[{ label: 'Predictions', value: fmtInt(row.count), color: 'var(--series-1)' }]} />
          }}
        />
        {thresholdBin >= 0 ? (
          <ReferenceLine
            x={bins[thresholdBin].bin}
            stroke="var(--ink-2)"
            strokeDasharray="4 3"
            label={{ value: `threshold ${threshold.toFixed(3)}`, position: 'top', fill: 'var(--ink-2)', fontSize: 12 }}
          />
        ) : null}
        <Bar dataKey="count" fill="var(--series-1)" maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function HistogramTable({ bins }: Props) {
  return (
    <DataTable
      rows={bins}
      rowKey={(r) => r.bin}
      empty="No predictions in this window"
      columns={[
        { key: 'bin', label: 'Probability', render: (r) => r.bin },
        { key: 'count', label: 'Predictions', num: true, render: (r) => fmtInt(r.count) },
      ]}
    />
  )
}
