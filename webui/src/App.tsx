import { useEffect, useRef, useState } from 'react'
import './App.css'
import { api } from './api'
import type { BBox, DisplayMode, FieldMapData, FieldQuad, Point, Session, SessionStatus } from './types'

const SELECTED_SESSION_STORAGE_KEY = 'fuel-density-map:selected-session'
const VIEWER_STORAGE_KEY = 'fuel-density-map:viewer'
const FIELD_ASSET_URL = '/assets/rebuilt-field.png'

function sortSessions(sessions: Session[]) {
  return [...sessions].sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
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

function formatTimestamp(value: string) {
  return new Intl.DateTimeFormat(undefined, {
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  }).format(new Date(value))
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
  if (typeof window === 'undefined') {
    return {
      mode: 'blend' as DisplayMode,
      overlayOpacity: 1,
    }
  }

  const stored = window.localStorage.getItem(VIEWER_STORAGE_KEY)
  if (!stored) {
    return {
      mode: 'blend' as DisplayMode,
      overlayOpacity: 1,
    }
  }

  try {
    const parsed = JSON.parse(stored) as { mode?: DisplayMode; overlayOpacity?: number }
    return {
      mode: parsed.mode ?? 'blend',
      overlayOpacity: parsed.overlayOpacity ?? 1,
    }
  } catch {
    return {
      mode: 'blend' as DisplayMode,
      overlayOpacity: 1,
    }
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

function drawFieldMapFrame(canvas: HTMLCanvasElement, fieldMapData: FieldMapData, frameIndex: number) {
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

  const r = FIELD_MAP_FUEL_DOT_RADIUS_PX
  for (const [normalizedX, normalizedY] of activeFrame) {
    const x = (normalizedX / 10000) * rect.width
    const y = (normalizedY / 10000) * rect.height

    context.beginPath()
    context.arc(x, y, r, 0, Math.PI * 2)
    context.fillStyle = 'rgba(245, 212, 62, 0.92)'
    context.shadowColor = 'rgba(245, 212, 62, 0.42)'
    context.shadowBlur = r * 1.8
    context.fill()
  }

  context.shadowBlur = 0
}

function App() {
  const viewerState = readInitialViewerState()
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
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isImporting, setIsImporting] = useState(false)
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
  const stageRef = useRef<HTMLDivElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const fieldCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const playbackFrameRef = useRef<number | null>(null)

  const selectedSession = sessions.find((session) => session.id === selectedSessionId) ?? null
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

  useEffect(() => {
    void (async () => {
      try {
        const loadedSessions = await api.listSessions()
        setSessions(sortSessions(loadedSessions))
      } catch (error) {
        setErrorMessage(error instanceof Error ? error.message : 'Unable to load saved sessions.')
      }
    })()
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
      }),
    )
  }, [mode, overlayOpacity])

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
    const render = () => drawFieldMapFrame(canvas, fieldMapData, overlayFrameIndex)
    render()

    const resizeObserver = new ResizeObserver(render)
    resizeObserver.observe(canvas)
    return () => {
      resizeObserver.disconnect()
    }
  }, [fieldMapData, isFieldMode, overlayFrameIndex])

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
      const imported = await api.importSession(youtubeUrl.trim())
      upsertSession(imported)
      setSelectedSessionId(imported.id)
      setYoutubeUrl('')
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unable to import the YouTube video.')
    } finally {
      setIsImporting(false)
    }
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
      setMode((currentMode) => (currentMode === 'field' ? 'field' : 'blend'))
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

  function handleStagePointerUp(event: React.PointerEvent<HTMLDivElement>) {
    if (isFieldMode || !selectedSession?.media.videoUrl) {
      return
    }

    const point = readPointFromPointerEvent(event, videoRef.current, stageRef.current?.getBoundingClientRect() ?? null)
    if (!point) {
      return
    }

    setErrorMessage(null)
    setDraftFieldPoints((current) => {
      const basePoints = current.length === 4 ? [] : current
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

  return (
    <div className="shell">
      <aside className="sidebar panel">
        <div className="sidebar-header">
          <p className="eyebrow">Fuel Density Map</p>
          <h1>Local overlay sessions for FRC Rebuilt</h1>
          <p className="lede">
            Import a YouTube clip, frame the field region you care about, then generate and review the density map
            without leaving the page.
          </p>
        </div>

        <form className="import-form" onSubmit={handleImport}>
          <label htmlFor="youtube-url">YouTube link</label>
          <textarea
            id="youtube-url"
            rows={3}
            value={youtubeUrl}
            onChange={(event) => setYoutubeUrl(event.target.value)}
            placeholder="https://www.youtube.com/watch?v=..."
          />
          <button className="primary-button" type="submit" disabled={isImporting}>
            {isImporting ? 'Importing...' : 'Import video'}
          </button>
        </form>

        <div className="session-list-header">
          <h2>Saved sessions</h2>
          <span>{sessions.length}</span>
        </div>

        <div className="session-list" role="list">
          {sessions.length === 0 ? (
            <div className="empty-state">
              <p>No local sessions yet.</p>
              <span>Imported videos are stored on disk and listed here automatically.</span>
            </div>
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
              <div className="session-card-top">
                <strong>{session.title}</strong>
                <span className={`status-pill status-${session.status}`}>{statusLabel(session.status)}</span>
              </div>
              <p>{formatDuration(session.video.duration)}</p>
              <small>Updated {formatTimestamp(session.updatedAt)}</small>
              <button
                className="delete-session-button"
                type="button"
                disabled={isDeleting}
                onClick={(event) => {
                  event.stopPropagation()
                  void handleDeleteSession(session.id)
                }}
              >
                Delete
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-column">
        <section className="hero-strip panel">
          <div>
            <p className="eyebrow">Workflow</p>
            <h2>Draw once, process locally, compare layers instantly</h2>
          </div>
          <div className="chip-row">
            <span className="chip">Single page</span>
            <span className="chip">Local sessions</span>
            <span className="chip">Overlay toggle</span>
            <span className="chip">Timeline scrub</span>
          </div>
        </section>

        {errorMessage ? <section className="message-strip error-strip">{errorMessage}</section> : null}

        <section className="workspace-grid">
          <div className="panel stage-panel">
            <div className="section-header">
              <div>
                <p className="eyebrow">Viewer</p>
                <h2>{selectedSession?.title ?? 'Choose or import a video session'}</h2>
              </div>
              {selectedSession ? (
                <div className="toolbar">
                  <div className="segmented-control">
                    {(['video', 'overlay', 'blend', 'field'] as DisplayMode[]).map((value) => (
                      <button
                        key={value}
                        className={mode === value ? 'segment-active' : ''}
                        onClick={() => setMode(value)}
                        type="button"
                      >
                        {value === 'video'
                          ? 'Video only'
                          : value === 'overlay'
                            ? 'Overlay only'
                            : value === 'blend'
                              ? 'Overlay on video'
                              : 'Field map'}
                      </button>
                    ))}
                  </div>

                  <button
                    className="primary-button"
                    onClick={() => void handleProcess()}
                    type="button"
                    disabled={
                      hasIncompleteFieldSelection ||
                      hasUnsavedFieldQuad ||
                      (!selectedSession.fieldQuad && !selectedSession.bbox) ||
                      isProcessing ||
                      isFieldQuadSaving ||
                      selectedSession.status === 'downloading'
                    }
                  >
                    {isProcessing || selectedSession.status === 'processing' ? 'Running...' : 'Run script'}
                  </button>
                </div>
              ) : null}
            </div>

            {selectedSession ? (
              <>
                <div className="stage">
                  <div
                    className={`stage-media ${isFieldMode ? 'field-stage' : ''}`}
                    ref={stageRef}
                    onPointerUp={handleStagePointerUp}
                    style={{
                      aspectRatio: isFieldMode
                        ? '3901 / 1583'
                        : selectedSession.video.width && selectedSession.video.height
                          ? `${selectedSession.video.width} / ${selectedSession.video.height}`
                          : '16 / 9',
                    }}
                  >
                    <video
                      key={selectedSession.id}
                      className={`stage-video ${isFieldMode ? 'stage-video-hidden' : ''}`}
                      ref={videoRef}
                      src={selectedSession.media.videoUrl ?? undefined}
                      preload="metadata"
                      playsInline
                      onLoadedMetadata={() => void handleLoadedMetadata()}
                      onDurationChange={handleVideoTimeUpdate}
                      onTimeUpdate={handleVideoTimeUpdate}
                      onPlay={handleVideoPlay}
                      onPause={handleVideoPause}
                      onEnded={handleVideoEnded}
                      onSeeked={handleVideoTimeUpdate}
                      style={{
                        opacity: mode === 'overlay' ? 0 : 1,
                      }}
                    />

                    {isFieldMode ? (
                      <>
                      <img alt="Top-down FRC Rebuilt field" className="stage-field-base" src={FIELD_ASSET_URL} />
                      <canvas className="stage-field-overlay" ref={fieldCanvasRef} />

                      {!fieldMapData && !isFieldMapLoading ? (
                        <div className="stage-empty-overlay">
                          <p>Run the script to generate projected fuel positions.</p>
                        </div>
                      ) : null}

                      {isFieldMapLoading ? (
                        <div className="stage-empty-overlay">
                          <p>Loading field map...</p>
                        </div>
                      ) : null}
                      </>
                    ) : (
                      <>
                      {overlayFrameUrl ? (
                        <img
                          alt="Fuel density overlay frame"
                          className="stage-overlay"
                          src={cacheBustUrl(overlayFrameUrl, selectedSession.updatedAt)}
                          style={{
                            opacity: mode === 'video' ? 0 : mode === 'overlay' ? 1 : overlayOpacity,
                          }}
                        />
                      ) : selectedSession.media.overlayTransparentUrl ? (
                        <img
                          alt="Fuel density overlay"
                          className="stage-overlay"
                          src={cacheBustUrl(
                            selectedSession.media.overlayTransparentUrl,
                            selectedSession.updatedAt,
                          )}
                          style={{
                            opacity: mode === 'video' ? 0 : mode === 'overlay' ? 1 : overlayOpacity,
                          }}
                        />
                      ) : null}

                      <div className="stage-drawing-layer" />

                      {activeFieldPoints.length > 0 ? (
                        <div className="field-selection-overlay">
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
                        <div className="stage-empty-overlay">
                          <p>Import a video to begin.</p>
                        </div>
                      ) : null}
                      </>
                    )}
                  </div>
                </div>

                <div className="viewer-controls">
                  <button className="ghost-button" onClick={togglePlayback} type="button">
                    {isPlaying ? 'Pause' : 'Play'}
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
                    />
                    <span>{formatDuration(duration)}</span>
                  </div>

                  <div className="opacity-control">
                    <label htmlFor="overlay-opacity">Overlay</label>
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
                        (!selectedSession.media.overlayTransparentUrl && !selectedSession.media.overlayFrameUrlTemplate)
                      }
                    />
                  </div>
                </div>

                <div className="helper-row">
                  <p>
                    {isFieldMode
                      ? 'Field map mode follows the current timeline frame. Switch back to the video to redefine the field borders.'
                      : activeFieldPoints.length > 0 && activeFieldPoints.length < 4
                        ? `Click ${4 - activeFieldPoints.length} more field corner${4 - activeFieldPoints.length === 1 ? '' : 's'} to finish the selection.`
                        : 'Click the four field corners on the video. The same Run script action updates both the overlay and the field map.'}
                  </p>
                  <div className="helper-actions">
                    <button
                      className="ghost-button"
                      onClick={() => {
                        setDraftFieldPoints((current) => (current.length > 0 && current.length < 4 ? current.slice(0, -1) : current))
                      }}
                      type="button"
                      disabled={activeFieldPoints.length === 0 || activeFieldPoints.length === 4}
                    >
                      Undo point
                    </button>
                    <button
                      className="ghost-button"
                      onClick={() => {
                        setDraftFieldPoints([])
                        void saveFieldQuad(null)
                      }}
                      type="button"
                    >
                      Clear field
                    </button>
                  </div>
                </div>
              </>
            ) : (
              <div className="stage-placeholder">
                <p>Paste a YouTube link to create the first local session.</p>
              </div>
            )}
          </div>

          <div className="meta-column">
            <section className="panel meta-panel">
              <p className="eyebrow">Session</p>
              <h2>{selectedSession?.title ?? 'No active session'}</h2>
              {selectedSession ? (
                <div className="stats-grid">
                  <div>
                    <span>Status</span>
                    <strong>{statusLabel(selectedSession.status)}</strong>
                  </div>
                  <div>
                    <span>Duration</span>
                    <strong>{formatDuration(selectedSession.video.duration)}</strong>
                  </div>
                  <div>
                    <span>Resolution</span>
                    <strong>
                      {selectedSession.video.width ?? '--'} x {selectedSession.video.height ?? '--'}
                    </strong>
                  </div>
                  <div>
                    <span>Last updated</span>
                    <strong>{formatTimestamp(selectedSession.updatedAt)}</strong>
                  </div>
                </div>
              ) : (
                <p className="muted-copy">Your imported sessions will appear here with their local analysis state.</p>
              )}
            </section>

            <section className="panel meta-panel">
              <p className="eyebrow">Field Selection</p>
              <h2>Projected borders</h2>
              {activePixelBox ? (
                <div className="stats-grid compact">
                  <div>
                    <span>Corners</span>
                    <strong>{activeCornerCount} / 4</strong>
                  </div>
                  <div>
                    <span>X</span>
                    <strong>{activePixelBox.x}px</strong>
                  </div>
                  <div>
                    <span>Y</span>
                    <strong>{activePixelBox.y}px</strong>
                  </div>
                  <div>
                    <span>Width</span>
                    <strong>{activePixelBox.width}px</strong>
                  </div>
                  <div>
                    <span>Height</span>
                    <strong>{activePixelBox.height}px</strong>
                  </div>
                </div>
              ) : (
                <p className="muted-copy">Click the four field corners in the video to define the projection and overlay crop.</p>
              )}
            </section>

            <section className="panel meta-panel">
              <p className="eyebrow">Overlay output</p>
              <h2>Generated heatmap stats</h2>
              {selectedSession?.overlay ? (
                <div className="stats-grid compact">
                  <div>
                    <span>Max value</span>
                    <strong>{selectedSession.overlay.stats.maxValue}</strong>
                  </div>
                  <div>
                    <span>Average</span>
                    <strong>{selectedSession.overlay.stats.actualAverage.toFixed(1)}</strong>
                  </div>
                  <div>
                    <span>Weighted avg</span>
                    <strong>{selectedSession.overlay.stats.weightedAverage.toFixed(1)}</strong>
                  </div>
                  <div>
                    <span>Non-zero pixels</span>
                    <strong>{selectedSession.overlay.stats.nonZeroPixels.toLocaleString()}</strong>
                  </div>
                </div>
              ) : (
                <p className="muted-copy">Run the script after setting a box to generate the local overlay assets.</p>
              )}
            </section>
          </div>
        </section>
      </main>
    </div>
  )
}

export default App
