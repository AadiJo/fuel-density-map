import { useEffect, useRef, useState } from 'react'
import './App.css'
import { api } from './api'
import type { BBox, DisplayMode, Session, SessionStatus } from './types'

const SELECTED_SESSION_STORAGE_KEY = 'fuel-density-map:selected-session'
const VIEWER_STORAGE_KEY = 'fuel-density-map:viewer'

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

function makeBox(start: { x: number; y: number }, end: { x: number; y: number }): BBox {
  const x = Math.min(start.x, end.x)
  const y = Math.min(start.y, end.y)
  return {
    x,
    y,
    width: Math.abs(end.x - start.x),
    height: Math.abs(end.y - start.y),
  }
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
  const [draftBox, setDraftBox] = useState<BBox | null>(null)
  const [isDrawing, setIsDrawing] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isScrubbing, setIsScrubbing] = useState(false)
  const [overlayFrameIndex, setOverlayFrameIndex] = useState(0)
  const [isDeleting, setIsDeleting] = useState(false)
  const stageRef = useRef<HTMLDivElement | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const pointerOriginRef = useRef<{ x: number; y: number } | null>(null)
  const playbackFrameRef = useRef<number | null>(null)

  const selectedSession = sessions.find((session) => session.id === selectedSessionId) ?? null
  const activeBox = draftBox ?? selectedSession?.bbox ?? null
  const activePixelBox = boxToPixels(activeBox, selectedSession)
  const overlayFrameUrl = buildOverlayFrameUrl(selectedSession?.media.overlayFrameUrlTemplate ?? null, overlayFrameIndex)

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
    setDraftBox(selectedSession?.bbox ?? null)
    setCurrentTime(0)
    setDuration(selectedSession?.video.duration ?? 0)
    setIsPlaying(false)
    setOverlayFrameIndex(0)
  }, [selectedSession?.id, selectedSession?.bbox, selectedSession?.video.duration])

  useEffect(() => {
    return () => {
      if (playbackFrameRef.current !== null) {
        window.cancelAnimationFrame(playbackFrameRef.current)
      }
    }
  }, [])

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

  function readPointFromEvent(event: React.PointerEvent<HTMLDivElement>) {
    const rect = stageRef.current?.getBoundingClientRect()
    if (!rect) {
      return null
    }

    return {
      x: clamp((event.clientX - rect.left) / rect.width),
      y: clamp((event.clientY - rect.top) / rect.height),
    }
  }

  async function saveBox(box: BBox | null) {
    if (!selectedSession) {
      return
    }

    try {
      setErrorMessage(null)
      const updated = await api.saveBBox(selectedSession.id, box)
      setDraftBox(updated.bbox)
      upsertSession(updated)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Unable to save the bounding box.')
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
      const processed = await api.processSession(selectedSession.id)
      upsertSession(processed)
      setMode('blend')
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

  function handleStagePointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (!selectedSession?.media.videoUrl) {
      return
    }

    const point = readPointFromEvent(event)
    if (!point) {
      return
    }

    pointerOriginRef.current = point
    setIsDrawing(true)
    setDraftBox({
      x: point.x,
      y: point.y,
      width: 0,
      height: 0,
    })
    event.currentTarget.setPointerCapture(event.pointerId)
  }

  function handleStagePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (!isDrawing || !pointerOriginRef.current) {
      return
    }

    const point = readPointFromEvent(event)
    if (!point) {
      return
    }

    setDraftBox(makeBox(pointerOriginRef.current, point))
  }

  function handleStagePointerUp(event: React.PointerEvent<HTMLDivElement>) {
    if (!isDrawing || !pointerOriginRef.current) {
      return
    }

    const point = readPointFromEvent(event)
    const start = pointerOriginRef.current
    setIsDrawing(false)
    pointerOriginRef.current = null
    event.currentTarget.releasePointerCapture(event.pointerId)

    if (!point) {
      return
    }

    const nextBox = makeBox(start, point)
    if (nextBox.width < 0.01 || nextBox.height < 0.01) {
      setDraftBox(selectedSession?.bbox ?? null)
      return
    }

    setDraftBox(nextBox)
    void saveBox(nextBox)
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
                    {(['video', 'overlay', 'blend'] as DisplayMode[]).map((value) => (
                      <button
                        key={value}
                        className={mode === value ? 'segment-active' : ''}
                        onClick={() => setMode(value)}
                        type="button"
                      >
                        {value === 'video' ? 'Video only' : value === 'overlay' ? 'Overlay only' : 'Overlay on video'}
                      </button>
                    ))}
                  </div>

                  <button
                    className="primary-button"
                    onClick={() => void handleProcess()}
                    type="button"
                    disabled={!selectedSession.bbox || isProcessing || selectedSession.status === 'downloading'}
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
                    className="stage-media"
                    ref={stageRef}
                    onPointerDown={handleStagePointerDown}
                    onPointerMove={handleStagePointerMove}
                    onPointerUp={handleStagePointerUp}
                    style={{
                      aspectRatio:
                        selectedSession.video.width && selectedSession.video.height
                          ? `${selectedSession.video.width} / ${selectedSession.video.height}`
                          : '16 / 9',
                    }}
                  >
                    <video
                      key={selectedSession.id}
                      className="stage-video"
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

                    {overlayFrameUrl ? (
                      <img
                        alt="Fuel density overlay frame"
                        className="stage-overlay"
                        src={overlayFrameUrl}
                        style={{
                          opacity: mode === 'video' ? 0 : mode === 'overlay' ? 1 : overlayOpacity,
                        }}
                      />
                    ) : selectedSession.media.overlayTransparentUrl ? (
                      <img
                        alt="Fuel density overlay"
                        className="stage-overlay"
                        src={selectedSession.media.overlayTransparentUrl}
                        style={{
                          opacity: mode === 'video' ? 0 : mode === 'overlay' ? 1 : overlayOpacity,
                        }}
                      />
                    ) : null}

                    <div className="stage-drawing-layer" />

                    {activeBox ? (
                      <div
                        className={`selection-box ${isDrawing ? 'selection-box-live' : ''}`}
                        style={{
                          left: `${activeBox.x * 100}%`,
                          top: `${activeBox.y * 100}%`,
                          width: `${activeBox.width * 100}%`,
                          height: `${activeBox.height * 100}%`,
                        }}
                      />
                    ) : null}

                    {!selectedSession.media.videoUrl ? (
                      <div className="stage-empty-overlay">
                        <p>Import a video to begin.</p>
                      </div>
                    ) : null}
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
                      disabled={!selectedSession.media.overlayTransparentUrl && !selectedSession.media.overlayFrameUrlTemplate}
                    />
                  </div>
                </div>

                <div className="helper-row">
                  <p>Drag directly over the frame to redraw the analysis region.</p>
                  <button className="ghost-button" onClick={() => void saveBox(null)} type="button">
                    Clear box
                  </button>
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
              <p className="eyebrow">Bounding box</p>
              <h2>Analysis region</h2>
              {activePixelBox ? (
                <div className="stats-grid compact">
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
                <p className="muted-copy">Draw a box on the viewer to limit where the overlay accumulates.</p>
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
