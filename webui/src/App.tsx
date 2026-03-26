import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import './App.css'
import { api } from './api'
import {
  IconActivity,
  IconAlert,
  IconBlend,
  IconCrosshair,
  IconDownload,
  IconEye,
  IconEyeOff,
  IconFilm,
  IconGauge,
  IconLayers,
  IconMap,
  IconPause,
  IconPlay,
  IconSidebarLeftToggle,
  IconSidebarRightToggle,
  IconTerminal,
  IconTrash,
  IconUndo,
  IconVideo,
  IconXCircle,
} from './icons'
import type {
  BBox,
  DisplayMode,
  FieldMapData,
  FieldQuad,
  Point,
  ProcessingProgress,
  Session,
  SessionStatus,
} from './types'

const SELECTED_SESSION_STORAGE_KEY = 'fuel-density-map:selected-session'
const VIEWER_STORAGE_KEY = 'fuel-density-map:viewer'
const FIELD_ASSET_URL = '/assets/rebuilt-field.png'

/** Normalized 0–1 coords on the full field PNG; must match processor_cli.py FIELD_DESTINATION_BOUNDS. */
const FIELD_IMAGE_NORM_BOUNDS = {
  minX: 0.133,
  maxX: 0.866,
  minY: 0.053,
  maxY: 0.946,
} as const

const FIELD_FUEL_EXCLUSION_ZONES = [
  [
    { x: 0.2812, y: 0.4991 },
    { x: 0.2966, y: 0.4302 },
    { x: 0.3302, y: 0.4302 },
    { x: 0.3461, y: 0.4991 },
    { x: 0.3302, y: 0.566 },
    { x: 0.2966, y: 0.566 },
  ],
  [
    { x: 0.6178, y: 0.4991 },
    { x: 0.6337, y: 0.4302 },
    { x: 0.6675, y: 0.4302 },
    { x: 0.6834, y: 0.4991 },
    { x: 0.6675, y: 0.566 },
    { x: 0.6337, y: 0.566 },
  ],
] as const

function sortSessions(sessions: Session[]) {
  return [...sessions].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
}

/** Merge server list with in-memory sessions so a slow initial GET cannot wipe a just-imported session. */
function mergeSessionListsWithServer(server: Session[], prev: Session[]): Session[] {
  const map = new Map<string, Session>()
  for (const s of prev) {
    map.set(s.id, s)
  }
  for (const s of server) {
    const cur = map.get(s.id)
    if (!cur || s.updatedAt >= cur.updatedAt) {
      map.set(s.id, s)
    }
  }
  return sortSessions([...map.values()])
}

function formatDuration(value: number | null | undefined) {
  if (!value || Number.isNaN(value) || value < 0) {
    return '0:00'
  }

  const totalSeconds = Math.floor(value)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

function clampTrimRange(start: number, end: number, durationSec: number) {
  const d = Math.max(0, durationSec)
  let s = Math.max(0, Math.min(start, d))
  let e = Math.max(0, Math.min(end, d))
  if (e - s < 0.05) {
    e = Math.min(d, s + 0.1)
  }
  if (e <= s) {
    e = Math.min(d, s + 0.1)
  }
  return { start: s, end: e }
}

function formatTrimKeepSpan(seconds: number) {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return '0s'
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`
  }
  return formatDuration(seconds)
}

function seekVideoTo(video: HTMLVideoElement | null, timeSec: number) {
  if (!video || !Number.isFinite(timeSec)) {
    return
  }
  const dur = video.duration
  if (Number.isFinite(dur) && dur > 0) {
    video.currentTime = Math.min(Math.max(0, timeSec), dur)
  } else {
    video.currentTime = Math.max(0, timeSec)
  }
}

function formatTimestamp(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value))
}

function formatElapsedSince(iso: string) {
  const sec = Math.max(0, Math.floor((Date.now() - new Date(iso).getTime()) / 1000))
  if (sec < 60) {
    return `${sec}s`
  }
  const m = Math.floor(sec / 60)
  const s = sec % 60
  return `${m}m ${s.toString().padStart(2, '0')}s`
}

function processingPhaseLabel(phase: string | undefined) {
  switch (phase) {
    case 'starting':
      return 'Starting'
    case 'analyze':
      return 'Analyzing video'
    case 'encode':
      return 'Saving & encoding'
    case 'frames':
      return 'Overlay frames'
    default:
      return phase ? phase : 'Working'
  }
}

/**
 * ETA from measured progress (avg rate since `startedAt`).
 * Falls back to a duration-based guess when progress is missing or `current` is still 0.
 * If the job runs past the first naive bound, we extend the estimate so the UI does not go blank.
 */
function estimateSecondsRemainingFromProgress(
  progress: ProcessingProgress | null | undefined,
  videoDurationSec: number | null | undefined,
  /** When `processingProgress` is missing (race / read error) but status is still processing */
  sessionUpdatedAtFallback?: string | null,
): number | null {
  /** Initial upper bound on total runtime (two-pass pipeline is often ~0.4–1.2× clip length). */
  const roughTotalBudgetSec = () => {
    if (videoDurationSec && videoDurationSec > 0) {
      return Math.max(60, Math.min(3600, videoDurationSec * 1.1))
    }
    return 240
  }

  const roughFromElapsed = (startedAtIso: string) => {
    const elapsedSec = (Date.now() - new Date(startedAtIso).getTime()) / 1000
    if (elapsedSec < 0.5) {
      return null
    }
    const budget = roughTotalBudgetSec()
    let remaining = budget - elapsedSec
    if (remaining <= 0) {
      // Still processing past the first guess — extrapolate from elapsed so we never return 0 (clock = "now").
      remaining = Math.max(90, elapsedSec * 0.4)
    }
    return Math.max(1, Math.round(remaining))
  }

  if (!progress) {
    if (!sessionUpdatedAtFallback) {
      return null
    }
    return roughFromElapsed(sessionUpdatedAtFallback)
  }
  if (progress.total <= 0) {
    return null
  }
  const { current, total, startedAt } = progress
  const elapsedSec = (Date.now() - new Date(startedAt).getTime()) / 1000

  if (current <= 0) {
    return roughFromElapsed(startedAt)
  }

  if (elapsedSec < 0.5) {
    return null
  }
  // Progress bar full but Python may still be writing stats / closing files.
  if (current >= total) {
    return 30
  }
  const rawRemaining = elapsedSec * ((total - current) / current)
  if (rawRemaining <= 0) {
    return 30
  }
  return Math.max(1, Math.ceil(rawRemaining))
}

/** Local clock time when processing is expected to finish; `null` seconds → placeholder. */
function formatEstimatedFinishTime(secondsRemaining: number | null) {
  if (secondsRemaining == null || secondsRemaining <= 0) {
    return '…'
  }
  const finish = new Date(Date.now() + secondsRemaining * 1000)
  return finish.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  })
}

function statusLabel(status: SessionStatus) {
  switch (status) {
    case 'completed':
      return 'Ready'
    case 'processing':
      return 'Processing'
    case 'downloading':
      return 'Downloading'
    case 'error':
      return 'Error'
    case 'ready':
      return 'Awaiting run'
    default:
      return 'Idle'
  }
}

function clamp(value: number, min = 0, max = 1) {
  return Math.min(Math.max(value, min), max)
}

function pointInPolygon(x: number, y: number, polygon: readonly Point[]) {
  let inside = false
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x
    const yi = polygon[i].y
    const xj = polygon[j].x
    const yj = polygon[j].y
    const intersects =
      yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / ((yj - yi) || Number.EPSILON) + xi
    if (intersects) {
      inside = !inside
    }
  }
  return inside
}

function pointIsInFieldFuelExclusionZone(x: number, y: number) {
  return FIELD_FUEL_EXCLUSION_ZONES.some((polygon) => pointInPolygon(x, y, polygon))
}

function boxToPixels(box: BBox | null, session: Session | null) {
  if (!box || !session?.video.width || !session.video.height) {
    return null
  }

  return {
    x: Math.round(box.x * session.video.width),
    y: Math.round(box.y * session.video.height),
    width: Math.round(box.width * session.video.width),
    height: Math.round(box.height * session.video.height),
  }
}

function readInitialViewerState() {
  const defaults = {
    mode: 'match' as DisplayMode,
    overlayOpacity: 1,
    showVideoLayer: true,
    showOverlayLayer: true,
  }

  if (typeof window === 'undefined') {
    return defaults
  }

  const stored = window.localStorage.getItem(VIEWER_STORAGE_KEY)
  if (!stored) {
    return defaults
  }

  try {
    const parsed = JSON.parse(stored) as {
      mode?: string
      overlayOpacity?: number
      showVideoLayer?: boolean
      showOverlayLayer?: boolean
    }

    const legacy = parsed.mode
    const mode: DisplayMode = legacy === 'field' ? 'field' : 'match'

    let showVideoLayer = true
    let showOverlayLayer = true

    if (parsed.showVideoLayer !== undefined || parsed.showOverlayLayer !== undefined) {
      showVideoLayer = parsed.showVideoLayer ?? true
      showOverlayLayer = parsed.showOverlayLayer ?? true
    } else if (legacy === 'video') {
      showVideoLayer = true
      showOverlayLayer = false
    } else if (legacy === 'overlay') {
      showVideoLayer = false
      showOverlayLayer = true
    }

    return {
      mode,
      overlayOpacity: parsed.overlayOpacity ?? 1,
      showVideoLayer,
      showOverlayLayer,
    }
  } catch {
    return defaults
  }
}

function buildOverlayFrameUrl(template: string | null, frameIndex: number) {
  if (!template) {
    return null
  }

  return template.replace('__FRAME__', String(frameIndex))
}

/** Bust browser cache after re-running the processor (same URL, new file on disk). */
function cacheBustUrl(url: string, version: string | null | undefined) {
  if (!version) {
    return url
  }
  const sep = url.includes('?') ? '&' : '?'
  return `${url}${sep}v=${encodeURIComponent(version)}`
}

/** Maps four clicked corners to [TL, TR, BR, BL] for OpenCV / processor_cli destination order. */
function orderQuadPoints(points: Point[]): FieldQuad | null {
  if (points.length !== 4) {
    return null
  }

  const sums = points.map((point) => point.x + point.y)
  const diffs = points.map((point) => point.y - point.x)
  const tlIdx = sums.indexOf(Math.min(...sums))
  const brIdx = sums.indexOf(Math.max(...sums))
  const trIdx = diffs.indexOf(Math.min(...diffs))
  const blIdx = diffs.indexOf(Math.max(...diffs))

  if (new Set([tlIdx, trIdx, brIdx, blIdx]).size === 4) {
    return [points[tlIdx], points[trIdx], points[brIdx], points[blIdx]]
  }

  const sortedByY = [...points].sort((left, right) => {
    if (left.y !== right.y) {
      return left.y - right.y
    }
    return left.x - right.x
  })

  const topRow = sortedByY.slice(0, 2).sort((left, right) => left.x - right.x)
  const bottomRow = sortedByY.slice(2).sort((left, right) => left.x - right.x)
  if (topRow.length !== 2 || bottomRow.length !== 2) {
    return null
  }

  const [topLeft, topRight] = topRow
  const [bottomLeft, bottomRight] = bottomRow
  return [topLeft, topRight, bottomRight, bottomLeft]
}

const NEAR_POINT_THRESH = 0.045

function findNearestPointIndex(points: Point[], p: Point): number | null {
  let best: number | null = null
  let bestD = NEAR_POINT_THRESH * NEAR_POINT_THRESH
  points.forEach((pt, index) => {
    const dx = pt.x - p.x
    const dy = pt.y - p.y
    const d = dx * dx + dy * dy
    if (d <= bestD) {
      bestD = d
      best = index
    }
  })
  return best
}

/** Inset and size of an image with `object-fit: contain` within a fixed box (CSS px). */
function computeContainedImageRect(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
): { offsetX: number; offsetY: number; dw: number; dh: number } | null {
  if (!containerWidth || !containerHeight || imageWidth <= 0 || imageHeight <= 0) {
    return null
  }
  const scale = Math.min(containerWidth / imageWidth, containerHeight / imageHeight)
  const dw = imageWidth * scale
  const dh = imageHeight * scale
  const offsetX = (containerWidth - dw) / 2
  const offsetY = (containerHeight - dh) / 2
  return { offsetX, offsetY, dw, dh }
}

/** Letterboxed video picture as % of stage (so overlays align with actual pixels). */
function computeVideoPictureLayoutPct(stage: HTMLElement, video: HTMLVideoElement) {
  const sRect = stage.getBoundingClientRect()
  const vw = video.videoWidth
  const vh = video.videoHeight
  const inset = computeContainedImageRect(sRect.width, sRect.height, vw, vh)
  if (!inset) {
    return null
  }
  const { offsetX: ox, offsetY: oy, dw, dh } = inset

  return {
    left: (ox / sRect.width) * 100,
    top: (oy / sRect.height) * 100,
    width: (dw / sRect.width) * 100,
    height: (dh / sRect.height) * 100,
  }
}

function readPointFromPointerEvent(
  event: React.PointerEvent<HTMLDivElement>,
  video: HTMLVideoElement | null,
  fallbackRect: DOMRect | null,
): Point | null {
  const vw = video?.videoWidth ?? 0
  const vh = video?.videoHeight ?? 0
  const vRect = video?.getBoundingClientRect()
  const rect = vRect && vRect.width > 0 && vRect.height > 0 ? vRect : fallbackRect
  if (!rect) {
    return null
  }

  if (vw > 0 && vh > 0) {
    const scale = Math.min(rect.width / vw, rect.height / vh)
    const displayedWidth = vw * scale
    const displayedHeight = vh * scale
    const offsetX = (rect.width - displayedWidth) / 2
    const offsetY = (rect.height - displayedHeight) / 2
    const x = (event.clientX - rect.left - offsetX) / displayedWidth
    const y = (event.clientY - rect.top - offsetY) / displayedHeight
    return { x: clamp(x), y: clamp(y) }
  }

  return {
    x: clamp((event.clientX - rect.left) / rect.width),
    y: clamp((event.clientY - rect.top) / rect.height),
  }
}

function quadToPolygonValue(points: Point[]) {
  return points.map((point) => `${point.x * 100},${point.y * 100}`).join(' ')
}

function fieldQuadsEqual(a: FieldQuad | null | undefined, b: FieldQuad | null): boolean {
  if (!a || !b) {
    return false
  }
  const eps = 1e-8
  return a.every(
    (point, index) =>
      Math.abs(point.x - b[index].x) < eps && Math.abs(point.y - b[index].y) < eps,
  )
}

/** CSS px; field map uses one marker size (third tuple value in JSON is legacy). */
const FIELD_MAP_FUEL_DOT_RADIUS_PX = 6

function drawFieldMapFrame(
  canvas: HTMLCanvasElement,
  fieldMapData: FieldMapData,
  frameIndex: number,
  fieldImage: HTMLImageElement | null,
) {
  const rect = canvas.getBoundingClientRect()
  if (!rect.width || !rect.height) {
    return
  }

  const context = canvas.getContext('2d')
  if (!context) {
    return
  }

  const devicePixelRatio = window.devicePixelRatio || 1
  const width = Math.round(rect.width * devicePixelRatio)
  const height = Math.round(rect.height * devicePixelRatio)

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width
    canvas.height = height
  }

  context.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)
  context.clearRect(0, 0, rect.width, rect.height)

  const maxFrameIndex = Math.max(fieldMapData.frames.length - 1, 0)
  const activeFrame = fieldMapData.frames[Math.min(frameIndex, maxFrameIndex)] ?? []

  // Match `object-fit: contain` on `.stage-field-base`: use the displayed asset's
  // intrinsic size when loaded. If JSON width/height differ from the PNG (e.g. stale
  // field-map.json), height can look fine while horizontal dw is wrong.
  const nw = fieldImage?.naturalWidth ?? 0
  const nh = fieldImage?.naturalHeight ?? 0
  const iw = nw > 0 && nh > 0 ? nw : fieldMapData.imageWidth
  const ih = nw > 0 && nh > 0 ? nh : fieldMapData.imageHeight
  const inset = computeContainedImageRect(rect.width, rect.height, iw, ih)

  const r = FIELD_MAP_FUEL_DOT_RADIUS_PX
  for (const [normalizedX, normalizedY] of activeFrame) {
    let fx = normalizedX / 10000
    let fy = normalizedY / 10000
    // Projection targets an inset quad on the asset, not 0..1 of the full bitmap; values
    // below the inset (e.g. nx→0 from clipped px) must not map into the PNG margins.
    fx = clamp(fx, FIELD_IMAGE_NORM_BOUNDS.minX, FIELD_IMAGE_NORM_BOUNDS.maxX)
    fy = clamp(fy, FIELD_IMAGE_NORM_BOUNDS.minY, FIELD_IMAGE_NORM_BOUNDS.maxY)
    if (pointIsInFieldFuelExclusionZone(fx, fy)) {
      continue
    }

    let x: number
    let y: number
    if (inset) {
      const { offsetX, offsetY, dw, dh } = inset
      x = offsetX + fx * dw
      y = offsetY + fy * dh
      x = clamp(x, offsetX + r, offsetX + dw - r)
      y = clamp(y, offsetY + r, offsetY + dh - r)
    } else {
      x = fx * rect.width
      y = fy * rect.height
      x = clamp(x, r, rect.width - r)
      y = clamp(y, r, rect.height - r)
    }

    const glow = context.createRadialGradient(x, y, r * 0.2, x, y, r * 2.6)
    glow.addColorStop(0, 'rgba(255, 232, 110, 0.62)')
    glow.addColorStop(0.45, 'rgba(245, 212, 62, 0.34)')
    glow.addColorStop(1, 'rgba(245, 212, 62, 0)')

    context.beginPath()
    context.arc(x, y, r * 2.6, 0, Math.PI * 2)
    context.fillStyle = glow
    context.fill()

    context.beginPath()
    context.arc(x, y, r, 0, Math.PI * 2)
    context.fillStyle = 'rgba(224, 175, 34, 0.96)'
    context.shadowColor = 'rgba(245, 212, 62, 0.32)'
    context.shadowBlur = r * 0.95
    context.fill()
  }

  context.shadowBlur = 0
}

function App() {
  const viewerState = readInitialViewerState()
  const [appReady, setAppReady] = useState(false)
  const [sessions, setSessions] = useState<Session[]>([])
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(() => {
    if (typeof window === 'undefined') {
      return null
    }
    return window.localStorage.getItem(SELECTED_SESSION_STORAGE_KEY)
  })
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [mode, setMode] = useState<DisplayMode>(viewerState.mode)
  const [overlayOpacity, setOverlayOpacity] = useState(viewerState.overlayOpacity)
  const [showVideoLayer, setShowVideoLayer] = useState(viewerState.showVideoLayer)
  const [showOverlayLayer, setShowOverlayLayer] = useState(viewerState.showOverlayLayer)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [inspectorOpen, setInspectorOpen] = useState(true)
  const [isImporting, setIsImporting] = useState(false)
  const [trimImportSession, setTrimImportSession] = useState<Session | null>(null)
  const [trimStartSec, setTrimStartSec] = useState(0)
  const [trimEndSec, setTrimEndSec] = useState(0)
  const [isTrimming, setIsTrimming] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [draftFieldPoints, setDraftFieldPoints] = useState<Point[]>([])
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [overlayFrameIndex, setOverlayFrameIndex] = useState(0)
  const [isDeleting, setIsDeleting] = useState(false)
  const [isFieldQuadSaving, setIsFieldQuadSaving] = useState(false)
  const [fieldMapData, setFieldMapData] = useState<FieldMapData | null>(null)
  const [isFieldMapLoading, setIsFieldMapLoading] = useState(false)
  const [processLogOpen, setProcessLogOpen] = useState(false)
  const [processLogText, setProcessLogText] = useState('')
  const [layersPopoverOpen, setLayersPopoverOpen] = useState(false)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const layersPopoverRef = useRef<HTMLDivElement | null>(null)
  const processLogPreRef = useRef<HTMLPreElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const trimPreviewVideoRef = useRef<HTMLVideoElement | null>(null)
  const fieldImageRef = useRef<HTMLImageElement | null>(null)
  const fieldCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const [fieldImageLayoutTick, setFieldImageLayoutTick] = useState(0)
  const playbackFrameRef = useRef<number | null>(null)
  const dragTargetRef = useRef<{ index: number } | null>(null)
  const dragMovedRef = useRef(false)
  const pointerDownRef = useRef<{ clientX: number; clientY: number; hitIndex: number | null } | null>(null)

  const [pictureLayout, setPictureLayout] = useState<{
    left: number
    top: number
    width: number
    height: number
  } | null>(null)
  const [etaTick, setEtaTick] = useState(0)

  const selectedSession = sessions.find((session) => session.id === selectedSessionId) ?? null
  /** Two `<video>` elements decoding the same URL (stage + trim modal) reliably blacks out the whole tab on some Windows/GPU stacks — only one may be active. */
  const hideStageVideoForTrim =
    trimImportSession != null &&
    Boolean(trimImportSession.media?.videoUrl) &&
    selectedSession != null &&
    selectedSession.id === trimImportSession.id
  const trimImportDurationSec = trimImportSession?.video.duration ?? 0
  const trimTimelineLeftPct =
    trimImportDurationSec > 0 ? (trimStartSec / trimImportDurationSec) * 100 : 0
  const trimTimelineWidthPct =
    trimImportDurationSec > 0
      ? ((trimEndSec - trimStartSec) / trimImportDurationSec) * 100
      : 0
  const orderedDraftQuad =
    draftFieldPoints.length === 4 ? orderQuadPoints(draftFieldPoints) : null
  const hasUnsavedFieldQuad =
    draftFieldPoints.length === 4 &&
    orderedDraftQuad != null &&
    !fieldQuadsEqual(selectedSession?.fieldQuad, orderedDraftQuad)
  const activeFieldPoints = draftFieldPoints.length > 0 ? draftFieldPoints : selectedSession?.fieldQuad ?? []
  const activeCornerCount =
    draftFieldPoints.length > 0 && draftFieldPoints.length < 4 ? draftFieldPoints.length : selectedSession?.fieldQuad?.length ?? 0
  const hasIncompleteFieldSelection = draftFieldPoints.length > 0 && draftFieldPoints.length < 4
  const activePixelBox = hasIncompleteFieldSelection ? null : boxToPixels(selectedSession?.bbox ?? null, selectedSession)
  const overlayFrameUrl = buildOverlayFrameUrl(selectedSession?.media.overlayFrameUrlTemplate ?? null, overlayFrameIndex)
  const isFieldMode = mode === 'field'
  const isMatchMode = mode === 'match'
  const overlayOpacityRendered =
    !showOverlayLayer ? 0 : showVideoLayer ? overlayOpacity : 1
  const isProcessRunning = isProcessing || selectedSession?.status === 'processing'

  const processEtaSeconds = useMemo(
    () =>
      estimateSecondsRemainingFromProgress(
        selectedSession?.processingProgress,
        selectedSession?.video.duration ?? null,
        selectedSession?.status === 'processing' ? selectedSession.updatedAt : null,
      ),
    [
      selectedSession?.processingProgress,
      selectedSession?.video.duration,
      selectedSession?.status,
      selectedSession?.updatedAt,
      etaTick,
    ],
  )

  useEffect(() => {
    if (!isProcessRunning) {
      return
    }
    const id = window.setInterval(() => {
      setEtaTick((value) => value + 1)
    }, 1000)
    return () => window.clearInterval(id)
  }, [isProcessRunning])

  useEffect(() => {
    if (!processLogOpen || !selectedSessionId) {
      return
    }
    let cancelled = false
    const tick = async () => {
      try {
        const text = await api.getProcessLog(selectedSessionId)
        if (!cancelled) {
          setProcessLogText(text)
        }
      } catch (error) {
        if (!cancelled) {
          setProcessLogText(
            error instanceof Error ? error.message : '(Unable to load processor output.)',
          )
        }
      }
    }
    void tick()
    const id = window.setInterval(tick, 1000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [processLogOpen, selectedSessionId])

  useEffect(() => {
    const el = processLogPreRef.current
    if (!el) {
      return
    }
    el.scrollTop = el.scrollHeight
  }, [processLogText])

  useEffect(() => {
    if (!processLogOpen) {
      return
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setProcessLogOpen(false)
        setProcessLogText('')
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [processLogOpen])

  useEffect(() => {
    if (!trimImportSession) {
      return
    }
    const duration = trimImportSession.video.duration ?? 0
    setTrimStartSec(0)
    setTrimEndSec(Math.max(0, duration))
  }, [trimImportSession])

  useEffect(() => {
    if (trimImportSession && !trimImportSession.media?.videoUrl) {
      setTrimImportSession(null)
    }
  }, [trimImportSession])

  useEffect(() => {
    if (!trimImportSession) {
      return
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setTrimImportSession(null)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [trimImportSession])

  useEffect(() => {
    if (!isMatchMode) {
      setLayersPopoverOpen(false)
    }
  }, [isMatchMode])

  useEffect(() => {
    if (!layersPopoverOpen) {
      return
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setLayersPopoverOpen(false)
      }
    }
    const onMouseDown = (event: MouseEvent) => {
      const el = layersPopoverRef.current
      if (el && !el.contains(event.target as Node)) {
        setLayersPopoverOpen(false)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('mousedown', onMouseDown)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('mousedown', onMouseDown)
    }
  }, [layersPopoverOpen])

  useEffect(() => {
    const stage = stageRef.current
    const video = videoRef.current
    if (!stage || !video) {
      return
    }

    const updateLayout = () => {
      setPictureLayout(computeVideoPictureLayoutPct(stage, video))
    }

    updateLayout()
    const observer = new ResizeObserver(updateLayout)
    observer.observe(stage)
    video.addEventListener('loadedmetadata', updateLayout)
    window.addEventListener('resize', updateLayout)
    return () => {
      observer.disconnect()
      video.removeEventListener('loadedmetadata', updateLayout)
      window.removeEventListener('resize', updateLayout)
    }
  }, [selectedSession?.id, selectedSession?.video.width, selectedSession?.video.height])

  const hasProcessingSession = sessions.some((session) => session.status === 'processing')
  const shouldPollSessions = hasProcessingSession || isProcessing

  useEffect(() => {
    if (!shouldPollSessions) {
      return
    }
    const tick = () => {
      void api
        .listSessions()
        .then(setSessions)
        .catch(() => {
          /* ignore poll errors */
        })
      if (selectedSessionId) {
        void api
          .getSession(selectedSessionId)
          .then((session) => {
            setSessions((current) => {
              const filtered = current.filter((s) => s.id !== session.id)
              return sortSessions([session, ...filtered])
            })
          })
          .catch(() => {
            /* ignore poll errors */
          })
      }
    }
    void tick()
    const id = window.setInterval(tick, 500)
    return () => window.clearInterval(id)
  }, [shouldPollSessions, selectedSessionId])

  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const loadedSessions = await api.listSessions()
        if (!cancelled) {
          setSessions((prev) => mergeSessionListsWithServer(loadedSessions, prev))
        }
      } catch (error) {
        if (!cancelled) {
          setErrorMessage(error instanceof Error ? error.message : 'Unable to load saved sessions.')
        }
      } finally {
        if (!cancelled) {
          setAppReady(true)
        }
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!selectedSessionId && sessions[0]) {
      setSelectedSessionId(sessions[0].id)
      return
    }

    if (selectedSessionId && !sessions.some((session) => session.id === selectedSessionId)) {
      setSelectedSessionId(sessions[0]?.id ?? null)
    }
  }, [selectedSessionId, sessions])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    if (selectedSessionId) {
      window.localStorage.setItem(SELECTED_SESSION_STORAGE_KEY, selectedSessionId)
    }
  }, [selectedSessionId])

  useEffect(() => {
    if (typeof window === 'undefined') {
      return
    }

    window.localStorage.setItem(
      VIEWER_STORAGE_KEY,
      JSON.stringify({
        mode,
        overlayOpacity,
        showVideoLayer,
        showOverlayLayer,
      }),
    )
  }, [mode, overlayOpacity, showVideoLayer, showOverlayLayer])

  useEffect(() => {
    setDraftFieldPoints(selectedSession?.fieldQuad ? [...selectedSession.fieldQuad] : [])
    setCurrentTime(0)
    setDuration(selectedSession?.video.duration ?? 0)
    setIsPlaying(false)
    setOverlayFrameIndex(0)
  }, [selectedSession?.id])

  useEffect(() => {
    if (!selectedSession) {
      return
    }
    setDraftFieldPoints(selectedSession.fieldQuad ? [...selectedSession.fieldQuad] : [])
  }, [selectedSession?.id, selectedSession?.fieldQuad])

  useEffect(() => {
    if (selectedSession?.video.duration != null) {
      setDuration(selectedSession.video.duration)
    }
  }, [selectedSession?.video.duration])

  useEffect(() => {
    return () => {
      if (playbackFrameRef.current !== null) {
        window.cancelAnimationFrame(playbackFrameRef.current)
      }
    }
  }, [])

  useEffect(() => {
    const fieldMapDataUrl = selectedSession?.media.fieldMapDataUrl
    setFieldMapData(null)

    if (!fieldMapDataUrl) {
      setIsFieldMapLoading(false)
      return
    }

    let isActive = true

    void (async () => {
      try {
        setIsFieldMapLoading(true)
        const loadedFieldMapData = await api.getFieldMapData(fieldMapDataUrl)
        if (isActive) {
          setFieldMapData(loadedFieldMapData)
        }
      } catch (error) {
        if (isActive) {
          setErrorMessage(error instanceof Error ? error.message : 'Unable to load the field map data.')
        }
      } finally {
        if (isActive) {
          setIsFieldMapLoading(false)
        }
      }
    })()

    return () => {
      isActive = false
    }
  }, [selectedSession?.id, selectedSession?.media.fieldMapDataUrl, selectedSession?.updatedAt])

  useEffect(() => {
    if (!isFieldMode || !fieldMapData || !selectedSession?.overlay) {
      return
    }
    const video = videoRef.current
    if (!video) {
      return
    }
    const fps = selectedSession.overlay.stats.overlayFps || 30
    const frameCount = selectedSession.overlay.stats.overlayFrameCount || 0
    const nextFrame = Math.max(0, Math.min(frameCount - 1, Math.floor(video.currentTime * fps)))
    setOverlayFrameIndex(nextFrame)
  }, [isFieldMode, fieldMapData, selectedSession?.overlay])

  useEffect(() => {
    if (!isFieldMode || !fieldMapData || !fieldCanvasRef.current) {
      return
    }

    const canvas = fieldCanvasRef.current
    const render = () =>
      drawFieldMapFrame(canvas, fieldMapData, overlayFrameIndex, fieldImageRef.current)
    render()

    const resizeObserver = new ResizeObserver(render)
    resizeObserver.observe(canvas)
    return () => {
      resizeObserver.disconnect()
    }
  }, [fieldMapData, isFieldMode, overlayFrameIndex, fieldImageLayoutTick])

  useLayoutEffect(() => {
    if (!isFieldMode) {
      return
    }
    const img = fieldImageRef.current
    if (img?.complete && img.naturalWidth > 0) {
      setFieldImageLayoutTick((tick) => tick + 1)
    }
  }, [isFieldMode])

  function upsertSession(nextSession: Session) {
    setSessions((current) => {
      const filtered = current.filter((session) => session.id !== nextSession.id)
      return sortSessions([nextSession, ...filtered])
    })
  }

  function syncOverlayFrame(targetTime?: number) {
    const video = videoRef.current
    if (!video || !selectedSession?.overlay) {
      return
    }

    const nextTime = targetTime ?? video.currentTime
    const fps = selectedSession.overlay.stats.overlayFps || 30
    const frameCount = selectedSession.overlay.stats.overlayFrameCount || 0
    const nextFrame = Math.max(0, Math.min(frameCount - 1, Math.floor(nextTime * fps)))
    setOverlayFrameIndex(nextFrame)
  }

  function startPlaybackSync() {
    if (playbackFrameRef.current !== null) {
      window.cancelAnimationFrame(playbackFrameRef.current)
    }

    const tick = () => {
      const video = videoRef.current
      if (video) {
        const time = video.currentTime
        if (!isScrubbing) {
          setCurrentTime(time)
        }
        syncOverlayFrame(time)
      }
      playbackFrameRef.current = window.requestAnimationFrame(tick)
    }

    playbackFrameRef.current = window.requestAnimationFrame(tick)
  }

  function stopPlaybackSync() {
    if (playbackFrameRef.current !== null) {
      window.cancelAnimationFrame(playbackFrameRef.current)
      playbackFrameRef.current = null
    }
  }

  async function saveFieldQuad(fieldQuad: FieldQuad | null, sessionIdOverride?: string): Promise<boolean> {
    const sessionId = sessionIdOverride ?? selectedSession?.id
    if (!sessionId) {
      setErrorMessage('No session selected.')
      return false
    }

    try {
      setIsFieldQuadSaving(true)
      setErrorMessage(null)
      setDraftFieldPoints(fieldQuad ? [...fieldQuad] : [])
      await api.saveFieldQuad(sessionId, fieldQuad)
      const fresh = await api.getSession(sessionId)
      setDraftFieldPoints(fresh.fieldQuad ? [...fresh.fieldQuad] : [])
      upsertSession(fresh)
      return true
    } catch (error) {
      setDraftFieldPoints(fieldQuad ? [...fieldQuad] : [])
      setErrorMessage(error instanceof Error ? error.message : 'Unable to save the field borders.')
      return false
    } finally {
      setIsFieldQuadSaving(false)
    }
  }

  async function handleImport(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault()

    if (!youtubeUrl.trim()) {
      setErrorMessage('Paste a YouTube link first.')
      return
    }

    try {
      setIsImporting(true)
      setErrorMessage(null)
      const result = await api.importSession(youtubeUrl.trim())
      upsertSession(result.session)
      setSelectedSessionId(result.session.id)
      setYoutubeUrl('')
      if (result.videoJustDownloaded) {
        setTrimImportSession(result.session)
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unable to import the YouTube video.')
    } finally {
      setIsImporting(false)
    }
  }

  async function handleTrimApply() {
    if (!trimImportSession?.media.videoUrl) {
      setTrimImportSession(null)
      return
    }
    const duration = trimImportSession.video.duration ?? 0
    const { start, end } = clampTrimRange(trimStartSec, trimEndSec, duration)
    try {
      setIsTrimming(true)
      setErrorMessage(null)
      const updated = await api.trimVideo(trimImportSession.id, start, end)
      upsertSession(updated)
      setTrimImportSession(null)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Could not trim video.')
    } finally {
      setIsTrimming(false)
    }
  }

  function handleTrimSkip() {
    setTrimImportSession(null)
  }

  async function handleProcess() {
    if (!selectedSession) {
      return
    }

    try {
      setIsProcessing(true)
      setErrorMessage(null)
      if (draftFieldPoints.length === 4) {
        const ordered = orderQuadPoints(draftFieldPoints)
        if (ordered && !fieldQuadsEqual(selectedSession.fieldQuad, ordered)) {
          const saved = await saveFieldQuad(ordered, selectedSession.id)
          if (!saved) {
            return
          }
        }
      }
      const processed = await api.processSession(selectedSession.id)
      upsertSession(processed)
      setMode((currentMode) => (currentMode === 'field' ? 'field' : 'match'))
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unable to generate the overlay.')
    } finally {
      setIsProcessing(false)
    }
  }

  async function handleDeleteSession(sessionId: string) {
    try {
      setIsDeleting(true)
      setErrorMessage(null)
      await api.deleteSession(sessionId)
      setSessions((current) => current.filter((session) => session.id !== sessionId))
      if (selectedSessionId === sessionId) {
        const nextSession = sessions.find((session) => session.id !== sessionId)
        setSelectedSessionId(nextSession?.id ?? null)
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unable to delete the session.')
    } finally {
      setIsDeleting(false)
    }
  }

  function baseFieldPoints(): Point[] {
    return draftFieldPoints.length > 0 ? draftFieldPoints : selectedSession?.fieldQuad ?? []
  }

  function handleDrawingPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (isFieldMode || !selectedSession?.media.videoUrl) {
      return
    }

    const point = readPointFromPointerEvent(
      event,
      videoRef.current,
      stageRef.current?.getBoundingClientRect() ?? null,
    )
    if (!point) {
      return
    }

    const points = baseFieldPoints()
    const hitIndex = findNearestPointIndex(points, point)
    pointerDownRef.current = {
      clientX: event.clientX,
      clientY: event.clientY,
      hitIndex,
    }
    dragMovedRef.current = false

    if (hitIndex !== null) {
      dragTargetRef.current = { index: hitIndex }
      event.currentTarget.setPointerCapture(event.pointerId)
    }
  }

  function handleDrawingPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (isFieldMode || dragTargetRef.current === null) {
      return
    }

    const point = readPointFromPointerEvent(
      event,
      videoRef.current,
      stageRef.current?.getBoundingClientRect() ?? null,
    )
    if (!point) {
      return
    }

    dragMovedRef.current = true
    const index = dragTargetRef.current.index

    setDraftFieldPoints((current) => {
      const base = current.length > 0 ? current : selectedSession?.fieldQuad ?? []
      if (index >= base.length) {
        return current
      }
      const next = [...base]
      next[index] = point
      return next
    })
  }

  function handleDrawingPointerUp(event: React.PointerEvent<HTMLDivElement>) {
    if (isFieldMode || !selectedSession?.media.videoUrl) {
      return
    }

    const point = readPointFromPointerEvent(
      event,
      videoRef.current,
      stageRef.current?.getBoundingClientRect() ?? null,
    )
    const hadDragTarget = dragTargetRef.current !== null
    const down = pointerDownRef.current
    pointerDownRef.current = null
    dragTargetRef.current = null

    if (down === null && !hadDragTarget) {
      return
    }

    try {
      event.currentTarget.releasePointerCapture(event.pointerId)
    } catch {
      /* released */
    }

    const movedPx =
      down &&
      (Math.abs(event.clientX - down.clientX) > 4 || Math.abs(event.clientY - down.clientY) > 4)
    const significantMove = dragMovedRef.current || movedPx
    dragMovedRef.current = false

    if (down && event.shiftKey && down.hitIndex !== null && !significantMove) {
      setErrorMessage(null)
      const removeIdx = down.hitIndex
      setDraftFieldPoints((current) => {
        const base = current.length > 0 ? current : selectedSession?.fieldQuad ?? []
        if (removeIdx >= base.length) {
          return current
        }
        const next = base.filter((_, i) => i !== removeIdx)
        if (next.length < 4) {
          void saveFieldQuad(null)
        }
        return next
      })
      return
    }

    if (hadDragTarget) {
      setDraftFieldPoints((current) => {
        if (current.length !== 4) {
          return current
        }
        const ordered = orderQuadPoints(current)
        if (ordered) {
          void saveFieldQuad(ordered)
          return [...ordered]
        }
        return current
      })
      return
    }

    if (significantMove) {
      return
    }

    if (!point) {
      return
    }

    setErrorMessage(null)
    setDraftFieldPoints((current) => {
      const basePoints = current.length === 4 ? [] : current
      if (basePoints.length >= 4) {
        return current
      }
      const nextPoints = [...basePoints, point]

      if (nextPoints.length < 4) {
        return nextPoints
      }

      const ordered = orderQuadPoints(nextPoints)
      if (!ordered) {
        setErrorMessage('Pick four distinct field corners.')
        return []
      }

      void saveFieldQuad(ordered)
      return [...ordered]
    })
  }

  function togglePlayback() {
    const video = videoRef.current
    if (!video) {
      return
    }

    if (video.paused) {
      syncOverlayFrame()
      void video.play()
    } else {
      video.pause()
    }
  }

  function handleScrub(nextTime: number) {
    const video = videoRef.current
    if (!video) {
      return
    }

    video.currentTime = nextTime
    syncOverlayFrame(nextTime)
    setCurrentTime(nextTime)
  }

  async function handleLoadedMetadata() {
    const video = videoRef.current
    if (!video || !selectedSession) {
      return
    }

    if (
      selectedSession.video.width === video.videoWidth &&
      selectedSession.video.height === video.videoHeight &&
      selectedSession.video.duration === video.duration
    ) {
      return
    }

    try {
      const updated = await api.updateVideoMetadata(
        selectedSession.id,
        video.videoWidth,
        video.videoHeight,
        video.duration,
      )
      upsertSession(updated)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unable to save video metadata.')
    }
  }

  function handleVideoTimeUpdate() {
    const video = videoRef.current
    if (!video) {
      return
    }

    if (!isScrubbing) {
      setCurrentTime(video.currentTime)
    }
    setDuration(video.duration || selectedSession?.video.duration || 0)
    syncOverlayFrame(video.currentTime)
  }

  function handleVideoPlay() {
    setIsPlaying(true)
    startPlaybackSync()
  }

  function handleVideoPause() {
    setIsPlaying(false)
    stopPlaybackSync()
  }

  function handleVideoEnded() {
    setIsPlaying(false)
    stopPlaybackSync()
    syncOverlayFrame(videoRef.current?.currentTime ?? 0)
  }

  const dashboardClass = [
    'dashboard',
    appReady ? 'dashboard--ready' : '',
    sidebarOpen ? '' : 'dashboard--sidebar-collapsed',
    inspectorOpen ? '' : 'dashboard--inspector-collapsed',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <div className={dashboardClass}>
      <div className="dashboard-chrome-outline" aria-hidden>
        <div className="dashboard-chrome-outline__titlebar" />
        <div className="dashboard-chrome-outline__rail" />
        <div className="dashboard-chrome-outline__sidebar" />
        <div className="dashboard-chrome-outline__main" />
        <div className="dashboard-chrome-outline__inspector" />
        <div className="dashboard-chrome-outline__footer" />
      </div>

      <header className="dash-titlebar shell-reveal shell-reveal--titlebar">
        <div className="dash-brand">
          <img
            className="dash-logo-mark"
            src="/favicon.svg"
            alt=""
            aria-hidden
            width={18}
            height={18}
            decoding="async"
            draggable={false}
            title="Fuel Density Map"
          />
        </div>
        <div className="dash-titlebar-center">
          {selectedSession ? (
            <>
              <span
                className={`status-dot status-${selectedSession.status}`}
                title={statusLabel(selectedSession.status)}
              />
              <p className="dash-session-title" data-active title={selectedSession.title}>
                {selectedSession.title}
              </p>
            </>
          ) : (
            <p className="dash-session-title" data-active={false}>
              No session selected
            </p>
          )}
        </div>
        {errorMessage ? (
          <div className="dash-titlebar-toast" role="status">
            <IconAlert size={16} />
            <span>{errorMessage}</span>
            <button type="button" onClick={() => setErrorMessage(null)} aria-label="Dismiss error">
              <IconXCircle size={16} />
            </button>
          </div>
        ) : (
          <span aria-hidden style={{ width: 8 }} />
        )}
      </header>

      <div className="dash-workspace">
        <nav className="dash-rail shell-reveal shell-reveal--rail" aria-label="Workspace">
          <span className="rail-logo" title="Viewer">
            <IconFilm size={20} />
          </span>
          <button
            type="button"
            className="rail-btn"
            aria-label={sidebarOpen ? 'Hide sessions panel' : 'Show sessions panel'}
            aria-pressed={sidebarOpen}
            title={sidebarOpen ? 'Hide sessions (collapse left)' : 'Show sessions'}
            onClick={() => setSidebarOpen((open) => !open)}
          >
            <IconSidebarLeftToggle collapsed={!sidebarOpen} size={20} />
          </button>
          <button
            type="button"
            className="rail-btn"
            aria-label={inspectorOpen ? 'Hide inspector' : 'Show inspector'}
            aria-pressed={inspectorOpen}
            title={inspectorOpen ? 'Hide inspector (collapse right)' : 'Show inspector'}
            onClick={() => setInspectorOpen((open) => !open)}
          >
            <IconSidebarRightToggle collapsed={!inspectorOpen} size={20} />
          </button>
        </nav>

        <aside className="dash-sidebar shell-reveal shell-reveal--sidebar" aria-label="Sessions and import">
          <div className="dash-sidebar-inner">
            <form className="import-form" onSubmit={handleImport}>
              <div className="import-form__row">
                <input
                  id="youtube-url"
                  type="url"
                  name="youtube-url"
                  autoComplete="off"
                  value={youtubeUrl}
                  onChange={(event) => setYoutubeUrl(event.target.value)}
                  placeholder="YouTube URL"
                  aria-label="YouTube URL"
                />
                <button
                  className="icon-btn primary"
                  type="submit"
                  disabled={isImporting}
                  title={isImporting ? 'Importing' : 'Import video'}
                  aria-label={isImporting ? 'Importing' : 'Import video'}
                >
                  <IconDownload size={18} />
                </button>
              </div>
            </form>

            <div className="session-sidebar-head">
              <span>Sessions</span>
              <span className="session-count">{sessions.length}</span>
            </div>

            <div className="session-list" role="list">
              {sessions.length === 0 ? (
                <div className="empty-state">No sessions yet. Import a clip to begin.</div>
              ) : null}

              {sessions.map((session) => (
                <div
                  key={session.id}
                  className={`session-card ${selectedSessionId === session.id ? 'session-card-active' : ''}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => setSelectedSessionId(session.id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault()
                      setSelectedSessionId(session.id)
                    }
                  }}
                >
                  <div className="session-card-row">
                    <span
                      className={`status-dot status-${session.status}`}
                      title={statusLabel(session.status)}
                    />
                    <span className="session-card-title" title={session.title}>
                      {session.title}
                    </span>
                  </div>
                  <div className="session-card-meta">
                    {formatDuration(session.video.duration)} · {formatTimestamp(session.updatedAt)}
                  </div>
                  <button
                    className="session-delete"
                    type="button"
                    disabled={isDeleting}
                    title="Delete session"
                    aria-label="Delete session"
                    onClick={(event) => {
                      event.stopPropagation()
                      void handleDeleteSession(session.id)
                    }}
                  >
                    <IconTrash size={16} />
                  </button>
                </div>
              ))}
            </div>
          </div>
        </aside>

        <main className="dash-main shell-reveal shell-reveal--main">
          <div className="dash-main-inner">
            <div className="stage-panel">
              <div
                className={`stage-toolbar ${selectedSession ? '' : 'stage-toolbar--empty'}`}
              >
                {selectedSession ? (
                  <div className="stage-toolbar-actions">
                    <div
                      className={`mode-toggle ${isFieldMode ? 'mode-toggle--field' : ''}`}
                      role="toolbar"
                      aria-label="Viewer mode"
                    >
                      <span className="mode-toggle__glider" aria-hidden />
                      <button
                        type="button"
                        className="mode-toggle__segment"
                        aria-pressed={isMatchMode}
                        title="Match view — video and heat overlay with layer controls"
                        onClick={() => setMode('match')}
                      >
                        <IconVideo size={17} />
                        <span>Match</span>
                      </button>
                      <button
                        type="button"
                        className="mode-toggle__segment"
                        aria-pressed={isFieldMode}
                        title="Field map — projected density on the pitch"
                        onClick={() => setMode('field')}
                      >
                        <IconMap size={17} />
                        <span>Field</span>
                      </button>
                    </div>

                    <div className="layers-popover-wrap" ref={layersPopoverRef}>
                      <button
                        type="button"
                        className={`layers-menu-btn ${layersPopoverOpen && isMatchMode ? 'layers-menu-btn--open' : ''}`}
                        aria-expanded={layersPopoverOpen && isMatchMode}
                        aria-haspopup="dialog"
                        aria-controls="match-layers-popover"
                        disabled={!isMatchMode}
                        title={
                          isMatchMode
                            ? 'Layers — show or hide video and heat overlay'
                            : 'Layers — switch to Match view to adjust layers'
                        }
                        onClick={() => {
                          if (!isMatchMode) {
                            return
                          }
                          setLayersPopoverOpen((open) => !open)
                        }}
                      >
                        <IconLayers size={18} />
                      </button>
                      {layersPopoverOpen && isMatchMode ? (
                        <div
                          id="match-layers-popover"
                          className="layers-popover"
                          role="dialog"
                          aria-label="Layers"
                        >
                          <div className="layers-popover__head">Layers</div>
                          <p className="layers-popover__hint">Top layer draws on top.</p>
                          <div className="layers-popover__list">
                            <button
                              type="button"
                              className={`match-layer ${showOverlayLayer ? 'match-layer--on' : ''}`}
                              onClick={() => setShowOverlayLayer((v) => !v)}
                              title={showOverlayLayer ? 'Hide overlay' : 'Show overlay'}
                              aria-pressed={showOverlayLayer}
                              disabled={
                                !selectedSession.media.overlayTransparentUrl &&
                                !selectedSession.media.overlayFrameUrlTemplate
                              }
                            >
                              <span className="match-layer__eye" aria-hidden>
                                {showOverlayLayer ? <IconEye size={16} /> : <IconEyeOff size={16} />}
                              </span>
                              <span className="match-layer__thumb match-layer__thumb--overlay" />
                              <span className="match-layer__label">Overlay</span>
                            </button>
                            <button
                              type="button"
                              className={`match-layer ${showVideoLayer ? 'match-layer--on' : ''}`}
                              onClick={() => setShowVideoLayer((v) => !v)}
                              title={showVideoLayer ? 'Hide video' : 'Show video'}
                              aria-pressed={showVideoLayer}
                            >
                              <span className="match-layer__eye" aria-hidden>
                                {showVideoLayer ? <IconEye size={16} /> : <IconEyeOff size={16} />}
                              </span>
                              <span className="match-layer__thumb match-layer__thumb--video" />
                              <span className="match-layer__label">Video</span>
                            </button>
                          </div>
                        </div>
                      ) : null}
                    </div>

                    <button
                      type="button"
                      className="process-log-btn"
                      onClick={() => setProcessLogOpen(true)}
                      title="View processor terminal output (stdout / stderr)"
                      aria-label="View processor terminal output"
                    >
                      <IconActivity size={18} />
                    </button>

                    <button
                      className={`run-btn ${isProcessRunning ? 'run-btn--busy' : ''}`}
                      onClick={() => void handleProcess()}
                      type="button"
                      title={
                        isProcessRunning
                          ? `Processing · ~${processEtaSeconds ?? '…'}s left`
                          : 'Run processing'
                      }
                      aria-label={
                        isProcessRunning
                          ? `Processing, about ${processEtaSeconds ?? ''} seconds remaining`
                          : 'Run processing script'
                      }
                      disabled={
                        hasIncompleteFieldSelection ||
                        hasUnsavedFieldQuad ||
                        (!selectedSession.fieldQuad && !selectedSession.bbox) ||
                        isProcessing ||
                        isFieldQuadSaving ||
                        selectedSession.status === 'downloading'
                      }
                    >
                      {isProcessRunning ? (
                        <span className="run-btn__spinner" aria-hidden />
                      ) : (
                        <IconTerminal size={18} />
                      )}
                    </button>
                  </div>
                ) : null}
              </div>

              {selectedSession ? (
                <>
                  <div className="stage-frame">
                    <div className="stage">
                      <div
                        className={`stage-media ${isFieldMode ? 'field-stage' : ''}`}
                        ref={stageRef}
                        style={{
                          aspectRatio: isFieldMode
                            ? '3901 / 1583'
                            : selectedSession.video.width && selectedSession.video.height
                              ? `${selectedSession.video.width} / ${selectedSession.video.height}`
                              : '16 / 9',
                        }}
                      >
                          <video
                            key={`${selectedSession.id}-${hideStageVideoForTrim ? 'trim-modal' : 'stage'}`}
                            className={[
                              'stage-video',
                              isFieldMode ? 'stage-video-offscreen' : '',
                              !isFieldMode && !showVideoLayer ? 'stage-video-hidden' : '',
                            ]
                              .filter(Boolean)
                              .join(' ')}
                            ref={videoRef}
                            src={
                              hideStageVideoForTrim
                                ? undefined
                                : (selectedSession.media.videoUrl ?? undefined)
                            }
                            preload="metadata"
                            playsInline
                            onLoadedMetadata={() => void handleLoadedMetadata()}
                            onDurationChange={handleVideoTimeUpdate}
                            onTimeUpdate={handleVideoTimeUpdate}
                            onPlay={handleVideoPlay}
                            onPause={handleVideoPause}
                            onEnded={handleVideoEnded}
                            onSeeked={handleVideoTimeUpdate}
                          />

                          {isFieldMode ? (
                            <>
                              <img
                                alt=""
                                className="stage-field-base"
                                src={FIELD_ASSET_URL}
                                ref={fieldImageRef}
                                onLoad={() => setFieldImageLayoutTick((tick) => tick + 1)}
                              />
                              <canvas className="stage-field-overlay" ref={fieldCanvasRef} />

                              {!fieldMapData && !isFieldMapLoading ? (
                                <div className="stage-empty-overlay stage-empty-overlay--field">
                                  Run processing to render the field map.
                                </div>
                              ) : null}

                              {isFieldMapLoading ? (
                                <div className="stage-empty-overlay stage-empty-overlay--field">
                                  Loading field map…
                                </div>
                              ) : null}
                            </>
                          ) : (
                            <>
                              {overlayFrameUrl ? (
                                <img
                                  alt=""
                                  className="stage-overlay"
                                  src={cacheBustUrl(overlayFrameUrl, selectedSession.updatedAt)}
                                  style={{
                                    opacity: overlayOpacityRendered,
                                  }}
                                />
                              ) : selectedSession.media.overlayTransparentUrl ? (
                                <img
                                  alt=""
                                  className="stage-overlay"
                                  src={cacheBustUrl(
                                    selectedSession.media.overlayTransparentUrl,
                                    selectedSession.updatedAt,
                                  )}
                                  style={{
                                    opacity: overlayOpacityRendered,
                                  }}
                                />
                              ) : null}

                              <div
                                className="stage-drawing-layer"
                                onPointerDown={handleDrawingPointerDown}
                                onPointerMove={handleDrawingPointerMove}
                                onPointerUp={handleDrawingPointerUp}
                                onPointerCancel={handleDrawingPointerUp}
                              />

                              {activeFieldPoints.length > 0 ? (
                                <div
                                  className="field-selection-overlay"
                                  style={
                                    pictureLayout
                                      ? {
                                          left: `${pictureLayout.left}%`,
                                          top: `${pictureLayout.top}%`,
                                          width: `${pictureLayout.width}%`,
                                          height: `${pictureLayout.height}%`,
                                          right: 'auto',
                                          bottom: 'auto',
                                        }
                                      : undefined
                                  }
                                >
                                  {activeFieldPoints.length >= 2 ? (
                                    <svg className="field-selection-shape" viewBox="0 0 100 100" preserveAspectRatio="none">
                                      {activeFieldPoints.length >= 3 ? (
                                        <polygon
                                          className={`field-selection-polygon ${activeFieldPoints.length === 4 ? 'field-selection-complete' : ''}`}
                                          points={quadToPolygonValue(activeFieldPoints)}
                                        />
                                      ) : (
                                        <polyline
                                          className="field-selection-polygon"
                                          points={quadToPolygonValue(activeFieldPoints)}
                                        />
                                      )}
                                    </svg>
                                  ) : null}

                                  {activeFieldPoints.map((point, index) => (
                                    <div
                                      key={`${point.x}-${point.y}-${index}`}
                                      className={`field-selection-point ${activeFieldPoints.length === 4 ? 'field-selection-point-complete' : ''}`}
                                      style={{
                                        left: `${point.x * 100}%`,
                                        top: `${point.y * 100}%`,
                                      }}
                                    >
                                      <span>{index + 1}</span>
                                    </div>
                                  ))}
                                </div>
                              ) : null}

                              {!selectedSession.media.videoUrl ? (
                                <div className="stage-empty-overlay">Import a video to preview.</div>
                              ) : null}
                            </>
                          )}
                      </div>
                    </div>

                    <div className="transport">
                      <button
                        className="transport-play"
                        type="button"
                        onClick={togglePlayback}
                        title={isPlaying ? 'Pause' : 'Play'}
                        aria-label={isPlaying ? 'Pause' : 'Play'}
                      >
                        {isPlaying ? <IconPause size={18} /> : <IconPlay size={18} />}
                      </button>

                      <div className="timeline-block">
                        <span>{formatDuration(currentTime)}</span>
                        <input
                          type="range"
                          min={0}
                          max={duration || 0}
                          step={0.01}
                          value={Math.min(currentTime, duration || 0)}
                          onMouseDown={() => setIsScrubbing(true)}
                          onMouseUp={() => setIsScrubbing(false)}
                          onTouchStart={() => setIsScrubbing(true)}
                          onTouchEnd={() => setIsScrubbing(false)}
                          onInput={(event) => handleScrub(Number((event.target as HTMLInputElement).value))}
                          onChange={(event) => handleScrub(Number(event.target.value))}
                          aria-label="Seek timeline"
                        />
                        <span>{formatDuration(duration)}</span>
                      </div>

                      <div className="opacity-control">
                        <label htmlFor="overlay-opacity" title="Overlay opacity">
                          <IconBlend size={16} />
                        </label>
                        <input
                          id="overlay-opacity"
                          type="range"
                          min={0.1}
                          max={1}
                          step={0.05}
                          value={overlayOpacity}
                          onChange={(event) => setOverlayOpacity(Number(event.target.value))}
                          disabled={
                            isFieldMode ||
                            !showVideoLayer ||
                            !showOverlayLayer ||
                            (!selectedSession.media.overlayTransparentUrl &&
                              !selectedSession.media.overlayFrameUrlTemplate)
                          }
                          aria-label="Overlay opacity"
                        />
                      </div>
                    </div>
                  </div>

                  <div className="helper-row">
                    <p>
                      {isFieldMode
                        ? 'Field map follows the playhead. Switch to Match view to place or adjust corners.'
                        : activeFieldPoints.length > 0 && activeFieldPoints.length < 4
                          ? `${4 - activeFieldPoints.length} corner${4 - activeFieldPoints.length === 1 ? '' : 's'} left · Shift+click removes · drag to move`
                          : 'Click four corners · drag to adjust · Shift+click removes · Restart clears all'}
                    </p>
                    <div className="helper-actions">
                      <button
                        className="ghost-btn"
                        onClick={() => {
                          setDraftFieldPoints((current) =>
                            current.length > 0 && current.length < 4 ? current.slice(0, -1) : current,
                          )
                        }}
                        type="button"
                        disabled={activeFieldPoints.length === 0 || activeFieldPoints.length === 4}
                        title="Undo last corner"
                        aria-label="Undo last corner"
                      >
                        <IconUndo size={16} />
                      </button>
                      <button
                        className="ghost-btn"
                        onClick={() => {
                          setDraftFieldPoints([])
                          void saveFieldQuad(null)
                        }}
                        type="button"
                        title="Restart field selection"
                        aria-label="Restart field selection"
                      >
                        Restart
                      </button>
                    </div>
                  </div>
                </>
              ) : (
                <div className="stage-frame stage-frame--solo">
                  <div className="stage-placeholder">Import a YouTube URL to open the viewer.</div>
                </div>
              )}
            </div>
          </div>
        </main>

        <aside className="dash-inspector shell-reveal shell-reveal--inspector" aria-label="Session details">
          <div className="dash-inspector-inner">
            <section className="inspector-card">
              <div className="inspector-card__head">
                <IconActivity size={16} />
                <h3>Session</h3>
              </div>
              {selectedSession ? (
                <>
                <div className="stat-grid">
                  <div className="stat-cell">
                    <span>Status</span>
                    <strong>{statusLabel(selectedSession.status)}</strong>
                  </div>
                  {isProcessRunning ? (
                    <>
                      <div className="stat-cell stat-cell--full">
                        <span>Step</span>
                        <strong
                          title={
                            selectedSession.processingProgress?.phase ?? 'Waiting for first progress update'
                          }
                        >
                          {selectedSession.processingProgress ? (
                            <>
                              {Math.min(
                                100,
                                Math.round(
                                  (selectedSession.processingProgress.current /
                                    Math.max(1, selectedSession.processingProgress.total)) *
                                    100,
                                ),
                              )}
                              % ·{' '}
                              {processingPhaseLabel(selectedSession.processingProgress.phase)}
                            </>
                          ) : (
                            'Starting…'
                          )}
                        </strong>
                      </div>
                      <div className="stat-cell">
                        <span>Running</span>
                        <strong title="Wall clock time for this run">
                          {formatElapsedSince(
                            selectedSession.processingProgress?.startedAt ?? selectedSession.updatedAt,
                          )}
                        </strong>
                      </div>
                    </>
                  ) : null}
                  <div className="stat-cell">
                    <span>Duration</span>
                    <strong>{formatDuration(selectedSession.video.duration)}</strong>
                  </div>
                  <div className="stat-cell">
                    <span>Resolution</span>
                    <strong>
                      {selectedSession.video.width ?? '—'}×{selectedSession.video.height ?? '—'}
                    </strong>
                  </div>
                  <div className="stat-cell">
                    <span>Updated</span>
                    <strong>{formatTimestamp(selectedSession.updatedAt)}</strong>
                  </div>
                  <div className="stat-cell stat-cell--full">
                    <span>Est. finish</span>
                    <strong
                      {...(isProcessRunning && processEtaSeconds != null
                        ? {
                            'title': `About ${processEtaSeconds}s remaining (from measured progress)`,
                          }
                        : {})}
                    >
                      {isProcessRunning
                        ? processEtaSeconds != null
                          ? `~${formatEstimatedFinishTime(processEtaSeconds)}`
                          : '…'
                        : '—'}
                    </strong>
                  </div>
                </div>
                {isProcessRunning ? (
                  <p className="inspector-note">
                    Full-length videos scan every frame twice and can take a long time. Open Processor output
                    (toolbar activity icon) to confirm the script is still printing.
                  </p>
                ) : null}
                </>
              ) : (
                <p className="inspector-muted">Select a session to inspect metadata.</p>
              )}
            </section>

            <section className="inspector-card">
              <div className="inspector-card__head">
                <IconCrosshair size={16} />
                <h3>Field</h3>
              </div>
              {activePixelBox ? (
                <div className="stat-grid">
                  <div className="stat-cell">
                    <span>Corners</span>
                    <strong>{activeCornerCount} / 4</strong>
                  </div>
                  <div className="stat-cell">
                    <span>X</span>
                    <strong>{activePixelBox.x}px</strong>
                  </div>
                  <div className="stat-cell">
                    <span>Y</span>
                    <strong>{activePixelBox.y}px</strong>
                  </div>
                  <div className="stat-cell">
                    <span>W × H</span>
                    <strong>
                      {activePixelBox.width}×{activePixelBox.height}
                    </strong>
                  </div>
                </div>
              ) : (
                <p className="inspector-muted">Place four corners on the video to lock the projection.</p>
              )}
            </section>

            <section className="inspector-card">
              <div className="inspector-card__head">
                <IconGauge size={16} />
                <h3>Heatmap</h3>
              </div>
              {selectedSession?.overlay ? (
                <div className="stat-grid stat-grid--single">
                  <div className="stat-cell">
                    <span>Max</span>
                    <strong>{selectedSession.overlay.stats.maxValue}</strong>
                  </div>
                  <div className="stat-cell">
                    <span>Avg</span>
                    <strong>{selectedSession.overlay.stats.actualAverage.toFixed(1)}</strong>
                  </div>
                  <div className="stat-cell">
                    <span>Weighted</span>
                    <strong>{selectedSession.overlay.stats.weightedAverage.toFixed(1)}</strong>
                  </div>
                  <div className="stat-cell">
                    <span>Non-zero px</span>
                    <strong>{selectedSession.overlay.stats.nonZeroPixels.toLocaleString()}</strong>
                  </div>
                </div>
              ) : (
                <p className="inspector-muted">Run the script after corners are set to populate stats.</p>
              )}
            </section>
          </div>
        </aside>
      </div>

      <footer className="dash-statusbar shell-reveal shell-reveal--footer">
        <span>Fuel Density Map · local workspace</span>
        <span>
          {sessions.length} session{sessions.length === 1 ? '' : 's'}
          {selectedSession ? ` · ${formatDuration(currentTime)} / ${formatDuration(duration)}` : ''}
        </span>
      </footer>

      {trimImportSession && trimImportSession.media.videoUrl ? (
        <div
          className="trim-import-overlay"
          role="presentation"
          onClick={handleTrimSkip}
        >
          <div
            className="trim-import-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="trim-import-title"
            onClick={(event) => event.stopPropagation()}
          >
            <h2 id="trim-import-title" className="trim-import-dialog__head">
              Trim clip
            </h2>
            <p className="trim-import-dialog__hint">
              Drag the green (in) and blue (out) handles on the timeline. The preview seeks as you move them.
              {trimImportSession.video.duration != null ? (
                <>
                  {' '}
                  Full length: <strong>{formatDuration(trimImportSession.video.duration)}</strong> (
                  {trimImportSession.video.duration.toFixed(1)}s).
                </>
              ) : null}
            </p>
            <div className="trim-import-preview">
              <video
                ref={trimPreviewVideoRef}
                key={`${trimImportSession.id}-${trimImportSession.updatedAt}`}
                src={`${trimImportSession.media.videoUrl}${trimImportSession.media.videoUrl.includes('?') ? '&' : '?'}v=${encodeURIComponent(trimImportSession.updatedAt)}`}
                controls
                playsInline
                preload="metadata"
                disablePictureInPicture
              />
            </div>
            {trimImportDurationSec > 0 ? (
              <div className="trim-timeline">
                <div className="trim-timeline__meta" aria-hidden>
                  <span className="trim-timeline__meta-item">
                    <span className="trim-timeline__meta-label">In</span>
                    <strong>{formatDuration(trimStartSec)}</strong>
                  </span>
                  <span className="trim-timeline__meta-item trim-timeline__meta-item--center">
                    <span className="trim-timeline__meta-label">Keep</span>
                    <strong>{formatTrimKeepSpan(trimEndSec - trimStartSec)}</strong>
                  </span>
                  <span className="trim-timeline__meta-item trim-timeline__meta-item--end">
                    <span className="trim-timeline__meta-label">Out</span>
                    <strong>{formatDuration(trimEndSec)}</strong>
                  </span>
                </div>
                <div className="trim-timeline__inputs">
                  <div className="trim-timeline__rail" aria-hidden>
                    <div className="trim-timeline__rail-bg" />
                    <div
                      className="trim-timeline__rail-selected"
                      style={{ left: `${trimTimelineLeftPct}%`, width: `${trimTimelineWidthPct}%` }}
                    />
                  </div>
                  <input
                    type="range"
                    className="trim-timeline__range trim-timeline__range--start"
                    aria-label="Trim start (in point)"
                    min={0}
                    max={trimImportDurationSec}
                    step={0.05}
                    value={Number.isFinite(trimStartSec) ? trimStartSec : 0}
                    onChange={(event) => {
                      const v = Number(event.target.value)
                      const { start, end } = clampTrimRange(v, trimEndSec, trimImportDurationSec)
                      setTrimStartSec(start)
                      setTrimEndSec(end)
                      seekVideoTo(trimPreviewVideoRef.current, start)
                    }}
                  />
                  <input
                    type="range"
                    className="trim-timeline__range trim-timeline__range--end"
                    aria-label="Trim end (out point)"
                    min={0}
                    max={trimImportDurationSec}
                    step={0.05}
                    value={Number.isFinite(trimEndSec) ? trimEndSec : 0}
                    onChange={(event) => {
                      const v = Number(event.target.value)
                      const { start, end } = clampTrimRange(trimStartSec, v, trimImportDurationSec)
                      setTrimStartSec(start)
                      setTrimEndSec(end)
                      seekVideoTo(trimPreviewVideoRef.current, end)
                    }}
                  />
                </div>
              </div>
            ) : (
              <p className="trim-import-dialog__hint trim-import-dialog__hint--solo">
                Clip length isn&apos;t available yet — use Skip to keep the full file, or try again after metadata
                loads.
              </p>
            )}
            <div className="trim-import-presets">
              <button
                type="button"
                onClick={() => {
                  const d = trimImportSession.video.duration ?? 0
                  if (d > 5) {
                    setTrimStartSec(5)
                  }
                }}
              >
                Start at 5s
              </button>
              <button
                type="button"
                onClick={() => {
                  const d = trimImportSession.video.duration ?? 0
                  if (d > 5) {
                    setTrimEndSec(Math.max(0.2, d - 5))
                  }
                }}
              >
                End 5s early
              </button>
              <button
                type="button"
                onClick={() => {
                  setTrimStartSec(0)
                  setTrimEndSec(trimImportSession.video.duration ?? 0)
                }}
              >
                Full clip
              </button>
            </div>
            <div className="trim-import-actions">
              <button type="button" onClick={handleTrimSkip} disabled={isTrimming}>
                Skip — keep full file
              </button>
              <button
                type="button"
                className="trim-import-actions__primary"
                onClick={() => void handleTrimApply()}
                disabled={isTrimming}
              >
                {isTrimming ? 'Trimming…' : 'Apply trim'}
              </button>
            </div>
          </div>
        </div>
      ) : null}

      {processLogOpen && selectedSessionId ? (
        <div
          className="process-log-overlay"
          role="presentation"
          onClick={() => {
            setProcessLogOpen(false)
            setProcessLogText('')
          }}
        >
          <div
            className="process-log-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="process-log-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="process-log-dialog__header">
              <div>
                <h2 id="process-log-title" className="process-log-dialog__title">
                  Processor output
                </h2>
                <p className="process-log-dialog__meta">
                  {selectedSession?.status === 'processing'
                    ? 'Live · refreshes about once per second'
                    : 'Stdout and stderr from the last processor run'}
                </p>
              </div>
              <button
                type="button"
                className="process-log-dialog__close"
                title="Close"
                onClick={() => {
                  setProcessLogOpen(false)
                  setProcessLogText('')
                }}
                aria-label="Close processor output"
              >
                <IconXCircle size={18} />
              </button>
            </div>
            <pre className="process-log-pre" ref={processLogPreRef}>
              {processLogText ||
                (selectedSession?.status === 'processing'
                  ? 'Waiting for output…'
                  : '(No processor log yet — run processing to capture output.)')}
            </pre>
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default App
