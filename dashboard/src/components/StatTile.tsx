import type { ReactNode } from 'react'

interface Props {
  label: string
  value: ReactNode
  note?: ReactNode
  hero?: boolean
}

// Stat tile contract: label (sentence case), value (semibold, proportional figures), optional note.
export default function StatTile({ label, value, note, hero }: Props) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className={hero ? 'value hero' : 'value'}>{value}</div>
      {note ? <div className="note">{note}</div> : null}
    </div>
  )
}
