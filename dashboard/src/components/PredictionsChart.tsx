import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { HourBucket } from '../api'
import { fmtHour, fmtInt } from '../format'
import ChartTip from './ChartTip'
import DataTable from './DataTable'

interface Props {
  buckets: HourBucket[]
}

const axisTick = { fill: 'var(--muted)', fontSize: 12 }

// Predictions per hour, stacked by regime. Slot 1 (blue) is FD001, slot 2 (orange) the replayed
// FD002 regime; the 2px surface-colored stroke is the gap between stacked segments.
export function PredictionsChart({ buckets }: Props) {
  if (buckets.length === 0) return <div className="empty">No predictions in this window</div>
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={buckets} margin={{ top: 8, right: 8, bottom: 4, left: -8 }}>
        <CartesianGrid stroke="var(--grid)" vertical={false} />
        <XAxis dataKey="hour" tickFormatter={fmtHour} tick={axisTick} axisLine={{ stroke: 'var(--axis)' }} tickLine={false} minTickGap={24} />
        <YAxis tick={axisTick} axisLine={false} tickLine={false} allowDecimals={false} tickFormatter={(v: number) => fmtInt(v)} />
        <Tooltip
          cursor={{ fill: 'var(--series-1-wash)' }}
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null
            const row = payload[0].payload as HourBucket
            return (
              <ChartTip
                title={`${fmtHour(row.hour)}, ${fmtInt(row.total)} predictions`}
                rows={[
                  { label: 'FD001', value: fmtInt(row.fd001 ?? 0), color: 'var(--series-1)' },
                  { label: 'FD002', value: fmtInt(row.fd002 ?? 0), color: 'var(--series-2)' },
                ]}
              />
            )
          }}
        />
        <Bar dataKey="fd001" stackId="regime" fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} maxBarSize={24} isAnimationActive={false} />
        <Bar dataKey="fd002" stackId="regime" fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} maxBarSize={24} radius={[4, 4, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}

export function PredictionsTable({ buckets }: Props) {
  return (
    <DataTable
      rows={[...buckets].reverse()}
      rowKey={(r) => r.hour}
      empty="No predictions in this window"
      columns={[
        { key: 'hour', label: 'Hour', render: (r) => fmtHour(r.hour) },
        { key: 'fd001', label: 'FD001', num: true, render: (r) => fmtInt(r.fd001 ?? 0) },
        { key: 'fd002', label: 'FD002', num: true, render: (r) => fmtInt(r.fd002 ?? 0) },
        { key: 'total', label: 'Total', num: true, render: (r) => fmtInt(r.total) },
      ]}
    />
  )
}

export const regimeLegend = (
  <>
    <span><span className="key" style={{ background: 'var(--series-1)' }} />FD001</span>
    <span><span className="key" style={{ background: 'var(--series-2)' }} />FD002 (replayed regime)</span>
  </>
)
