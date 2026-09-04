// Small formatting helpers shared by tiles, charts, tooltips, and tables.

export const fmtInt = (n: number | null | undefined) => (n == null ? '–' : new Intl.NumberFormat().format(n))

export const fmtNum = (n: number | null | undefined, digits = 4) => (n == null ? '–' : n.toFixed(digits))

export const fmtPct = (n: number | null | undefined) => (n == null ? '–' : `${(n * 100).toFixed(1)}%`)

export const fmtTime = (iso: string | null | undefined) =>
  iso ? new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : '–'

export const fmtHour = (iso: string) => new Date(iso).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit' })

// Axis ticks on a time axis: minutes when the span is short, hours for a day or two, days beyond.
export const tickFormatter = (spanMs: number) => (t: number) => {
  const d = new Date(t)
  if (spanMs < 6 * 3600_000) return d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  if (spanMs < 3 * 86400_000) return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit' })
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

export const fmtPct0 = (n: number) => `${Math.round(n * 100)}%`

export const fmtDay = (iso: string) => new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })

export const regimeName = (regime: string) => (regime === 'fd002' ? 'FD002 (replayed regime)' : 'FD001')

export const seriesColor = (regime: string) => (regime === 'fd002' ? 'var(--series-2)' : 'var(--series-1)')
