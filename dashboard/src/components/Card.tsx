import { useState, type ReactNode } from 'react'

interface Props {
  title: string
  hint?: string
  chart: ReactNode
  table?: ReactNode
  stale?: boolean
  wide?: boolean
  legend?: ReactNode
}

// Every chart ships with a table twin behind a toggle, so no value is reachable only by hover.
// A card whose content is already a table gets no toggle. While a refetch is in flight the
// previous render is held at reduced opacity: no skeleton flash.
export default function Card({ title, hint, chart, table, stale, wide, legend }: Props) {
  const [view, setView] = useState<'chart' | 'table'>('chart')
  return (
    <section className={wide ? 'card wide' : 'card'} aria-label={title}>
      <header>
        <h2>{title}</h2>
        {table ? (
          <div className="toggle" role="group" aria-label={`${title}: view`}>
            <button type="button" aria-pressed={view === 'chart'} onClick={() => setView('chart')}>Chart</button>
            <button type="button" aria-pressed={view === 'table'} onClick={() => setView('table')}>Table</button>
          </div>
        ) : null}
      </header>
      {hint ? <p className="hint">{hint}</p> : null}
      {view === 'chart' && legend ? <div className="legend">{legend}</div> : null}
      <div className={stale ? 'body stale' : 'body'}>{view === 'chart' || !table ? chart : table}</div>
    </section>
  )
}
