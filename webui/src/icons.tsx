type IconProps = {
  className?: string
  size?: number
}

function strokeProps() {
  return {
    fill: 'none' as const,
    stroke: 'currentColor',
    strokeWidth: 1.65,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
  }
}

export function IconFilm({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M4 6.5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6.5Z" />
      <path {...strokeProps()} d="M16.5 9.5 20 7v10l-3.5-2.5" />
    </svg>
  )
}

export function IconClapper({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M4 11h16v8a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-8Z" />
      <path {...strokeProps()} d="M4 11V9l3-2 3 2 3-2 3 2 3-2v2" />
    </svg>
  )
}

/** Double chevron left — “collapse / hide toward the left” (sessions rail). */
export function IconChevronsLeft({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="m14 7-4 5 4 5M20 7l-4 5 4 5" />
    </svg>
  )
}

/** Double chevron right — “expand from the left” or “collapse toward the right” (inspector rail). */
export function IconChevronsRight({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="m10 7 4 5-4 5M4 7l4 5-4 5" />
    </svg>
  )
}

/**
 * Primary (left) sidebar: VS Code–style frame + solid vertical bar (sessions list).
 * `collapsed` true → bar slides right (“panel hidden”).
 */
export function IconSidebarLeftToggle({ className, size = 20, collapsed }: IconProps & { collapsed: boolean }) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <rect x="4" y="5" width="16" height="14" rx="2" fill="none" stroke="currentColor" strokeWidth="1.65" />
      <g className={collapsed ? 'sidebar-toggle-bar sidebar-toggle-bar--left-off' : 'sidebar-toggle-bar'}>
        <rect x="5.5" y="6.75" width="5" height="10.5" rx="1" className="sidebar-toggle-bar__fill" />
      </g>
    </svg>
  )
}

/**
 * Inspector (right): same outer frame, but icon is a text/list column — distinct from the left bar.
 * `collapsed` true → lines dim and nudge (panel hidden).
 */
export function IconSidebarRightToggle({ className, size = 20, collapsed }: IconProps & { collapsed: boolean }) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <rect x="4" y="5" width="16" height="14" rx="2" fill="none" stroke="currentColor" strokeWidth="1.65" />
      <g className={collapsed ? 'inspector-rail-lines inspector-rail-lines--collapsed' : 'inspector-rail-lines'}>
        <line x1="7" y1="8" x2="17" y2="8" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" />
        <line x1="7" y1="11.5" x2="14.5" y2="11.5" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" />
        <line x1="7" y1="15" x2="17" y2="15" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" />
      </g>
    </svg>
  )
}

export function IconEye({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M2 12s4-6 10-6 10 6 10 6-4 6-10 6S2 12 2 12Z" />
      <path {...strokeProps()} d="M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z" />
    </svg>
  )
}

export function IconEyeOff({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M3 3 21 21" />
      <path {...strokeProps()} d="M10.6 10.6a2 2 0 0 0 2.8 2.8" />
      <path {...strokeProps()} d="M9.9 5.1A18.5 18.5 0 0 1 12 5c6 0 10 7 10 7a18.8 18.8 0 0 1-5.1 5.2M6.3 6.3A18.5 18.5 0 0 0 2 12s4 7 10 7c1.1 0 2.1-.2 3.1-.5" />
    </svg>
  )
}

export function IconLink({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path
        {...strokeProps()}
        d="M10 13a5 5 0 0 1 0-7l1-1a5 5 0 0 1 7 7l-1 1M14 11a5 5 0 0 1 0 7l-1 1a5 5 0 0 1-7-7l1-1"
      />
    </svg>
  )
}

export function IconDownload({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M12 4v11" />
      <path {...strokeProps()} d="m8 12 4 4 4-4" />
      <path {...strokeProps()} d="M5 20h14" />
    </svg>
  )
}

export function IconPlay({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M8 5v14l11-7-11-7Z" />
    </svg>
  )
}

export function IconPause({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M8 5v14M16 5v14" />
    </svg>
  )
}

export function IconVideo({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M4 7.5A2.5 2.5 0 0 1 6.5 5h7A2.5 2.5 0 0 1 16 7.5v9A2.5 2.5 0 0 1 13.5 19h-7A2.5 2.5 0 0 1 4 16.5v-9Z" />
      <path {...strokeProps()} d="m17 9 3-2v10l-3-2" />
    </svg>
  )
}

export function IconLayers({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="m12 4 9 4-9 4-9-4 9-4Z" />
      <path {...strokeProps()} d="m3 10 9 4 9-4" />
      <path {...strokeProps()} d="m3 14 9 4 9-4" />
    </svg>
  )
}

export function IconBlend({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M4 16c4 0 6-4 8-8 2 4 4 8 8 8" />
      <path {...strokeProps()} d="M4 8c4 0 6 4 8 8 2-4 4-8 8-8" />
    </svg>
  )
}

export function IconMap({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M4 6 9 4v14l-5 2V6Z" />
      <path {...strokeProps()} d="m15 4 5 2v14l-5-2V4Z" />
      <path {...strokeProps()} d="M9 4v14l6-2V2" />
    </svg>
  )
}

export function IconTerminal({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M5 5h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2Z" />
      <path {...strokeProps()} d="m8 10 3 2-3 2" />
      <path {...strokeProps()} d="M13 14h3" />
    </svg>
  )
}

export function IconTrash({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M9 4h6l1 2h4v2H4V6h4l1-2Z" />
      <path {...strokeProps()} d="M7 10v9a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-9" />
      <path {...strokeProps()} d="M10 14v4M14 14v4" />
    </svg>
  )
}

export function IconUndo({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M9 14 4 9l5-5" />
      <path {...strokeProps()} d="M4 9h10.5a4.5 4.5 0 0 1 0 9H12" />
    </svg>
  )
}

export function IconXCircle({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Z" />
      <path {...strokeProps()} d="m15 9-6 6M9 9l6 6" />
    </svg>
  )
}

export function IconAlert({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M12 9v5" />
      <path {...strokeProps()} d="M12 17h.01" />
      <path {...strokeProps()} d="M10.3 4.7 3.2 17.3c-.5 1 .2 2.2 1.3 2.2h15c1.1 0 1.8-1.2 1.3-2.2L13.7 4.7c-.5-1-1.9-1-2.4 0Z" />
    </svg>
  )
}

export function IconActivity({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M4 12h4l2-7 4 14 2-7h4" />
    </svg>
  )
}

export function IconGauge({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M12 14v3" />
      <path {...strokeProps()} d="M5.2 18.2A9 9 0 0 1 4 13c0-4.4 3.1-8 7.2-8.9" />
      <path {...strokeProps()} d="M20 13a9 9 0 0 1-1 3.2" />
      <path {...strokeProps()} d="M9.2 4.1A9 9 0 0 1 20 13" />
    </svg>
  )
}

export function IconCrosshair({ className, size = 20 }: IconProps) {
  return (
    <svg className={className} width={size} height={size} viewBox="0 0 24 24" aria-hidden>
      <path {...strokeProps()} d="M12 3v3M12 18v3M3 12h3M18 12h3" />
      <path {...strokeProps()} d="M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z" />
    </svg>
  )
}
