import { useCallback, useEffect, useState } from 'react'

export type Theme = 'light' | 'dark'

// Mirrors vondraysanford.com/theme.js: the pre-paint snippet in index.html has already set
// data-theme; this keeps the toggle, the stored choice, the meta theme-color, and the OS
// preference in sync. Same localStorage key as the site, so the choice carries across.
const STORAGE_KEY = 'theme'
const META_COLOR: Record<Theme, string> = { light: '#F7F7F5', dark: '#121216' }

function current(): Theme {
  return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}

function stored(): Theme | null {
  try {
    const s = localStorage.getItem(STORAGE_KEY)
    return s === 'light' || s === 'dark' ? s : null
  } catch {
    return null
  }
}

function apply(theme: Theme) {
  document.documentElement.dataset.theme = theme
  const meta = document.querySelector<HTMLMetaElement>('meta[name="theme-color"]')
  if (meta) meta.content = META_COLOR[theme]
}

export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(current)

  const toggle = useCallback(() => {
    const next: Theme = current() === 'dark' ? 'light' : 'dark'
    try {
      localStorage.setItem(STORAGE_KEY, next)
    } catch {
      // private mode or blocked storage: the toggle still works for this page view
    }
    apply(next)
    setTheme(next)
  }, [])

  // Track the OS preference only while the visitor hasn't chosen explicitly.
  useEffect(() => {
    const mq = window.matchMedia('(prefers-color-scheme: dark)')
    const onChange = (e: MediaQueryListEvent) => {
      if (stored()) return
      const next: Theme = e.matches ? 'dark' : 'light'
      apply(next)
      setTheme(next)
    }
    mq.addEventListener('change', onChange)
    return () => mq.removeEventListener('change', onChange)
  }, [])

  return [theme, toggle]
}
