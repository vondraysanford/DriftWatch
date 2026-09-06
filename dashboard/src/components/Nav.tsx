import { useEffect, useRef, useState } from 'react'
import { useTheme } from '../theme'

// The portfolio site's nav, as the sibling projects carry it: the same links in the same order
// pointing back at vondraysanford.com, the logo naming where you are, the theme toggle, a
// hamburger under 1000px, and the 1px reading-progress hairline (current position is live state).
const SITE = 'https://vondraysanford.com'
const LINKS: Array<[label: string, anchor: string]> = [
  ['Projects', 'projects'],
  ['About', 'about'],
  ['Skills', 'skills'],
  ['Experience', 'experience'],
  ['Certs', 'certifications'],
  ['Lab', 'lab'],
  ['Writing', 'writing'],
  ['Contact', 'contact'],
]

export default function Nav() {
  const [open, setOpen] = useState(false)
  const [theme, toggleTheme] = useTheme()
  const nav = useRef<HTMLElement>(null)
  const progress = useRef<HTMLDivElement>(null)

  // Nav border + reading progress, rAF-throttled like the site's own handler.
  useEffect(() => {
    let ticking = false
    const onScroll = () => {
      const el = nav.current
      const bar = progress.current
      if (el) el.style.borderBottomColor = window.scrollY > 50 ? 'var(--hairline-strong)' : 'var(--hairline)'
      if (bar) {
        const max = document.documentElement.scrollHeight - window.innerHeight
        bar.style.transform = `scaleX(${max > 0 ? window.scrollY / max : 0})`
      }
      ticking = false
    }
    const onEvent = () => {
      if (ticking) return
      ticking = true
      window.requestAnimationFrame(onScroll)
    }
    window.addEventListener('scroll', onEvent, { passive: true })
    window.addEventListener('resize', onEvent, { passive: true })
    onScroll()
    return () => {
      window.removeEventListener('scroll', onEvent)
      window.removeEventListener('resize', onEvent)
    }
  }, [])

  return (
    <nav ref={nav} className="portfolio-nav">
      <div className="nav-inner">
        <a href={`${SITE}/`} className="logo">
          vondray<span>.sanford</span><span className="logo-here"> / driftwatch</span>
        </a>
        <ul className={open ? 'nav-links open' : 'nav-links'} id="navLinks">
          {LINKS.map(([label, anchor]) => (
            <li key={anchor}>
              <a href={`${SITE}/#${anchor}`} onClick={() => setOpen(false)}>{label}</a>
            </li>
          ))}
        </ul>
        <button
          className="theme-toggle"
          type="button"
          aria-label="Toggle dark mode"
          aria-pressed={theme === 'dark'}
          onClick={toggleTheme}
        >
          <svg className="icon-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /></svg>
          <svg className="icon-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" /></svg>
        </button>
        <button
          className="hamburger"
          type="button"
          aria-label="Toggle navigation"
          aria-expanded={open}
          aria-controls="navLinks"
          onClick={() => setOpen((o) => !o)}
        >
          <span /><span /><span />
        </button>
      </div>
      <div className="nav-progress" ref={progress} aria-hidden="true" />
    </nav>
  )
}
