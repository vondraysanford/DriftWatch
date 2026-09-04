import type { ReactNode } from 'react'

export interface TipRow {
  label: string
  value: ReactNode
  color?: string
}

interface Props {
  title: ReactNode
  rows: TipRow[]
}

// Tooltip body shared by every chart: a title line and label/value rows, text in text tokens
// with a colored key beside the label carrying series identity.
export default function ChartTip({ title, rows }: Props) {
  return (
    <div className="tip" role="status">
      <div className="t">{title}</div>
      {rows.map((r) => (
        <div className="row" key={r.label}>
          <span>
            {r.color ? <span className="key" style={{ background: r.color, display: 'inline-block', width: 10, height: 10, borderRadius: 3, marginRight: 6, verticalAlign: -1 }} /> : null}
            {r.label}
          </span>
          <span>{r.value}</span>
        </div>
      ))}
    </div>
  )
}
