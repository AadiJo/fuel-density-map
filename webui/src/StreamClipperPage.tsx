import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react'
import './StreamClipperPage.css'
import { api } from './api'
import { IconActivity, IconAlert, IconGauge, IconPause, IconPlay, IconVideo } from './icons'
import type { StreamClipperStatus } from './types'

const DEFAULT_URL = 'https://www.youtube.com/watch?v=WMCVEBcqu1k'

const EMPTY_STATUS: StreamClipperStatus = {
  running: false,
  phase: 'stopped',
  matchState: 'idle',
  streamUrl: DEFAULT_URL,
  resolvedStreamUrl: null,
  startedAt: null,
  updatedAt: new Date(0).toISOString(),
  lastError: null,
  workDir: '',
  videosDir: '',
  config: {
    preRollSec: 8,
    postRollSec: 8,
    rollingBufferSec: 45,
    segmentTimeSec: 2,
    startThresholdSec: 75,
    maxMatchSec: 180,
    stopGraceSec: 5,
    minTimerConfidence: 0.35,
    ocrRegion: {
      x: 0.34,
      y: 0.84,
      width: 0.32,
      height: 0.13,
    },
  },
  bufferedSeconds: 0,
  bufferedSegments: 0,
  latestDetection: null,
  recentClips: [],
  activeMatch: {
    startedAt: null,
    secondsLeft: null,
    statusText: 'Stopped',
  },
}

function formatSeconds(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return '--:--'
  }
  const total = Math.max(0, Math.floor(value))
  const minutes = Math.floor(total / 60)
  const seconds = total % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

function formatUpdatedAt(value: string) {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return 'waiting...'
  }
  return date.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
    second: '2-digit',
  })
}

function phaseLabel(status: StreamClipperStatus) {
  if (status.phase === 'error') {
    return 'Error'
  }
  if (!status.running) {
    return 'Stopped'
  }
  if (status.matchState === 'active') {
    return 'Match live'
  }
  if (status.phase === 'saving') {
    return 'Saving clip'
  }
  if (status.phase === 'buffering') {
    return 'Buffering'
  }
  return 'Watching stream'
}

export default function StreamClipperPage() {
  const calibrationRef = useRef<HTMLDivElement | null>(null)
  const [status, setStatus] = useState<StreamClipperStatus>(EMPTY_STATUS)
  const [loading, setLoading] = useState(true)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [configLoaded, setConfigLoaded] = useState(false)
  const [url, setUrl] = useState(DEFAULT_URL)
  const [preRollSec, setPreRollSec] = useState('8')
  const [postRollSec, setPostRollSec] = useState('8')
  const [rollingBufferSec, setRollingBufferSec] = useState('45')
  const [segmentTimeSec, setSegmentTimeSec] = useState('2')
  const [startThresholdSec, setStartThresholdSec] = useState('75')
  const [maxMatchSec, setMaxMatchSec] = useState('180')
  const [stopGraceSec, setStopGraceSec] = useState('5')
  const [minTimerConfidence, setMinTimerConfidence] = useState('0.35')
  const [ocrRegion, setOcrRegion] = useState(EMPTY_STATUS.config.ocrRegion)
  const [frameUrl, setFrameUrl] = useState('/api/stream-clipper/frame')
  const [frameError, setFrameError] = useState<string | null>(null)
  const [pendingCorner, setPendingCorner] = useState<{ x: number; y: number } | null>(null)

  useEffect(() => {
    document.documentElement.classList.add('stream-page-root')
    document.body.classList.add('stream-page-root')
    document.getElementById('root')?.classList.add('stream-page-root')

    return () => {
      document.documentElement.classList.remove('stream-page-root')
      document.body.classList.remove('stream-page-root')
      document.getElementById('root')?.classList.remove('stream-page-root')
    }
  }, [])

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        const nextStatus = await api.getStreamClipperStatus()
        if (cancelled) {
          return
        }
        setStatus(nextStatus)
        setError(null)
        if (!configLoaded) {
          setUrl(nextStatus.streamUrl ?? DEFAULT_URL)
          setPreRollSec(String(nextStatus.config.preRollSec))
          setPostRollSec(String(nextStatus.config.postRollSec))
          setRollingBufferSec(String(nextStatus.config.rollingBufferSec))
          setSegmentTimeSec(String(nextStatus.config.segmentTimeSec))
          setStartThresholdSec(String(nextStatus.config.startThresholdSec))
          setMaxMatchSec(String(nextStatus.config.maxMatchSec))
          setStopGraceSec(String(nextStatus.config.stopGraceSec))
          setMinTimerConfidence(String(nextStatus.config.minTimerConfidence))
          setOcrRegion(nextStatus.config.ocrRegion)
          setConfigLoaded(true)
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Could not load clipper status.')
        }
      } finally {
        if (!cancelled) {
          setLoading(false)
        }
      }
    }

    void load()
    const interval = window.setInterval(() => {
      void load()
    }, 2000)

    return () => {
      cancelled = true
      window.clearInterval(interval)
    }
  }, [configLoaded])

  const handleStart = async () => {
    setSubmitting(true)
    try {
      const nextStatus = await api.startStreamClipper({
        url,
        preRollSec: Number(preRollSec),
        postRollSec: Number(postRollSec),
        rollingBufferSec: Number(rollingBufferSec),
        segmentTimeSec: Number(segmentTimeSec),
        startThresholdSec: Number(startThresholdSec),
        maxMatchSec: Number(maxMatchSec),
        stopGraceSec: Number(stopGraceSec),
        minTimerConfidence: Number(minTimerConfidence),
        ocrRegion,
      })
      setStatus(nextStatus)
      setError(null)
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : 'Could not start saving clips.')
    } finally {
      setSubmitting(false)
    }
  }

  const handleStop = async () => {
    setSubmitting(true)
    try {
      const nextStatus = await api.stopStreamClipper()
      setStatus(nextStatus)
      setError(null)
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : 'Could not stop the clipper.')
    } finally {
      setSubmitting(false)
    }
  }

  const liveText =
    status.matchState === 'active' && status.latestDetection
      ? `Match playing (${status.latestDetection.text} left)`
      : status.activeMatch.statusText

  const handleCalibratePointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    const host = calibrationRef.current
    if (!host) {
      return
    }
    const bounds = host.getBoundingClientRect()
    const x = Math.min(Math.max((event.clientX - bounds.left) / bounds.width, 0), 1)
    const y = Math.min(Math.max((event.clientY - bounds.top) / bounds.height, 0), 1)

    if (!pendingCorner) {
      setPendingCorner({ x, y })
      return
    }
    setOcrRegion({
      x: Math.min(pendingCorner.x, x),
      y: Math.min(pendingCorner.y, y),
      width: Math.max(0.02, Math.abs(x - pendingCorner.x)),
      height: Math.max(0.02, Math.abs(y - pendingCorner.y)),
    })
    setPendingCorner(null)
  }

  const handleSaveRegion = async () => {
    setSubmitting(true)
    try {
      const nextStatus = await api.saveStreamClipperOcrRegion(ocrRegion)
      setStatus(nextStatus)
      setError(null)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Could not save OCR region.')
    } finally {
      setSubmitting(false)
    }
  }

  const refreshFrame = () => {
    setFrameError(null)
    setFrameUrl(`/api/stream-clipper/frame?ts=${Date.now()}`)
  }

  return (
    <main className="stream-page">
      <div className="stream-page__backdrop" />
      <header className="stream-page__header">
        <div>
          <p className="stream-page__eyebrow">Separate Route</p>
          <h1>Stream Splitter</h1>
          <p className="stream-page__lede">
            Live-match clipping on its own address, with the same house style as the main dashboard.
          </p>
        </div>
        <div className="stream-page__header-actions">
          <a className="stream-link-chip" href="/">
            Back to main UI
          </a>
          <div className={`stream-pill stream-pill--${status.phase === 'error' ? 'error' : status.running ? 'live' : 'idle'}`}>
            <IconActivity size={16} />
            <span>{phaseLabel(status)}</span>
          </div>
        </div>
      </header>

      <section className="stream-hero">
        <div className="stream-hero__status">
          <div className="stream-hero__icon-wrap">
            <IconVideo size={24} />
          </div>
          <div>
            <p className="stream-hero__label">Status</p>
            <h2>{liveText}</h2>
            <p className="stream-hero__meta">
              Last update {formatUpdatedAt(status.updatedAt)} • Buffer {Math.round(status.bufferedSeconds)}s
            </p>
          </div>
        </div>

        <div className="stream-hero__timer">
          <p className="stream-hero__label">Clock</p>
          <div className="stream-hero__time">{formatSeconds(status.latestDetection?.seconds)}</div>
          <p className="stream-hero__meta">
            OCR confidence {Math.round((status.latestDetection?.confidence ?? 0) * 100)}%
          </p>
        </div>
      </section>

      {(error || status.lastError) && (
        <section className="stream-alert" role="alert">
          <IconAlert size={18} />
          <span>{error ?? status.lastError}</span>
        </section>
      )}

      <section className="stream-grid">
        <article className="stream-card">
          <div className="stream-card__head">
            <div>
              <p className="stream-card__eyebrow">Capture Control</p>
              <h3>Recorder</h3>
            </div>
            <div className="stream-card__buttons">
              <button className="stream-btn stream-btn--primary" disabled={submitting || status.running} onClick={handleStart}>
                <IconPlay size={16} />
                <span>Start saving videos</span>
              </button>
              <button className="stream-btn" disabled={submitting || !status.running} onClick={handleStop}>
                <IconPause size={16} />
                <span>Stop saving videos</span>
              </button>
            </div>
          </div>

          <label className="stream-field">
            <span>Livestream URL</span>
            <input value={url} onChange={(event) => setUrl(event.target.value)} placeholder={DEFAULT_URL} />
          </label>

          <div className="stream-form-grid">
            <label className="stream-field">
              <span>Pre-roll (sec)</span>
              <input value={preRollSec} onChange={(event) => setPreRollSec(event.target.value)} inputMode="decimal" />
            </label>
            <label className="stream-field">
              <span>Post-roll (sec)</span>
              <input value={postRollSec} onChange={(event) => setPostRollSec(event.target.value)} inputMode="decimal" />
            </label>
            <label className="stream-field">
              <span>Rolling buffer (sec)</span>
              <input value={rollingBufferSec} onChange={(event) => setRollingBufferSec(event.target.value)} inputMode="decimal" />
            </label>
            <label className="stream-field">
              <span>Segment size (sec)</span>
              <input value={segmentTimeSec} onChange={(event) => setSegmentTimeSec(event.target.value)} inputMode="decimal" />
            </label>
            <label className="stream-field">
              <span>Start threshold (sec)</span>
              <input value={startThresholdSec} onChange={(event) => setStartThresholdSec(event.target.value)} inputMode="numeric" />
            </label>
            <label className="stream-field">
              <span>Max match timer (sec)</span>
              <input value={maxMatchSec} onChange={(event) => setMaxMatchSec(event.target.value)} inputMode="numeric" />
            </label>
            <label className="stream-field">
              <span>Stop grace (sec)</span>
              <input value={stopGraceSec} onChange={(event) => setStopGraceSec(event.target.value)} inputMode="decimal" />
            </label>
            <label className="stream-field">
              <span>Min OCR confidence</span>
              <input value={minTimerConfidence} onChange={(event) => setMinTimerConfidence(event.target.value)} inputMode="decimal" />
            </label>
          </div>
        </article>

        <article className="stream-card stream-card--side">
          <div className="stream-card__head">
            <div>
              <p className="stream-card__eyebrow">Telemetry</p>
              <h3>Live detection</h3>
            </div>
            <IconGauge size={18} />
          </div>

          <dl className="stream-stats">
            <div>
              <dt>Detector phase</dt>
              <dd>{phaseLabel(status)}</dd>
            </div>
            <div>
              <dt>Match state</dt>
              <dd>{status.matchState}</dd>
            </div>
            <div>
              <dt>Latest timer</dt>
              <dd>{status.latestDetection?.text ?? 'none yet'}</dd>
            </div>
            <div>
              <dt>OCR region</dt>
              <dd>{status.latestDetection?.region ?? 'configured'}</dd>
            </div>
            <div>
              <dt>Segments buffered</dt>
              <dd>{status.bufferedSegments}</dd>
            </div>
            <div>
              <dt>Page loaded</dt>
              <dd>{loading ? 'refreshing...' : 'live'}</dd>
            </div>
          </dl>
        </article>

        <article className="stream-card stream-card--wide">
          <div className="stream-card__head">
            <div>
              <p className="stream-card__eyebrow">Calibration</p>
              <h3>Timer region</h3>
            </div>
            <div className="stream-card__buttons">
              <button className="stream-btn" type="button" onClick={refreshFrame}>
                Refresh frame
              </button>
              <button className="stream-btn stream-btn--primary" type="button" disabled={submitting} onClick={handleSaveRegion}>
                Save OCR region
              </button>
            </div>
          </div>

          <p className="stream-empty">
            Draw a box directly around the bottom timer and scoreboard area. This keeps OCR fast and avoids scanning the whole broadcast.
          </p>

          <div className="stream-calibration">
            <div ref={calibrationRef} className="stream-calibration__frame" onPointerDown={handleCalibratePointer}>
              <img
                src={frameUrl}
                alt="Live stream calibration frame"
                draggable={false}
                onLoad={() => setFrameError(null)}
                onError={() => setFrameError('Could not load a calibration frame yet. Refresh after a few more buffered segments.')}
              />
              <div
                className="stream-calibration__region"
                style={{
                  left: `${ocrRegion.x * 100}%`,
                  top: `${ocrRegion.y * 100}%`,
                  width: `${ocrRegion.width * 100}%`,
                  height: `${ocrRegion.height * 100}%`,
                }}
              />
            </div>
            {frameError && <p className="stream-calibration__error">{frameError}</p>}
            {!frameError && (
              <p className="stream-calibration__hint">
                {pendingCorner
                  ? 'Click the opposite corner of the timer box.'
                  : 'Click one corner of the timer box, then click the opposite corner.'}
              </p>
            )}
            <div className="stream-calibration__meta">
              <span>X {ocrRegion.x.toFixed(3)}</span>
              <span>Y {ocrRegion.y.toFixed(3)}</span>
              <span>W {ocrRegion.width.toFixed(3)}</span>
              <span>H {ocrRegion.height.toFixed(3)}</span>
            </div>
          </div>
        </article>

        <article className="stream-card stream-card--wide">
          <div className="stream-card__head">
            <div>
              <p className="stream-card__eyebrow">Output</p>
              <h3>Saved clips</h3>
            </div>
          </div>

          {status.recentClips.length > 0 ? (
            <ul className="stream-clips">
              {status.recentClips.map((clip) => (
                <li key={clip}>
                  <a href={`/clip-videos/${encodeURIComponent(clip)}`} target="_blank" rel="noreferrer">
                    {clip}
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <p className="stream-empty">No clips saved yet. Once the timer logic finds a full match window, new files will land in `videos/`.</p>
          )}
        </article>
      </section>
    </main>
  )
}
