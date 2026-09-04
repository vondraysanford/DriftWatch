import type { ReactNode } from 'react'

export interface Column<Row> {
  key: string
  label: string
  num?: boolean
  render: (row: Row) => ReactNode
}

interface Props<Row> {
  columns: Column<Row>[]
  rows: Row[]
  empty?: string
  rowKey: (row: Row, index: number) => string
}

export default function DataTable<Row>({ columns, rows, empty = 'No data', rowKey }: Props<Row>) {
  if (rows.length === 0) return <div className="empty">{empty}</div>
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c.key} className={c.num ? 'num' : undefined}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={rowKey(row, i)}>
              {columns.map((c) => (
                <td key={c.key} className={c.num ? 'num' : undefined}>{c.render(row)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
