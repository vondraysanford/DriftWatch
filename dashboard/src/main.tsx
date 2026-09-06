import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import './styles.css'

// The color scheme is set before first paint by the inline script in index.html
// (stored choice, ?theme=dark|light override, or the OS setting); src/theme.ts keeps it in sync.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
