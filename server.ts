import { execSync, spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdir, readdir, readFile, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'

type SessionStatus = 'idle' | 'downloading' | 'ready' | 'processing' | 'completed' | 'error'

type BBox = {
  x: number
  y: number
  width: number
  height: number
}

type Point = {
  x: number
  y: number
}

type FieldQuad = [Point, Point, Point, Point]

type OverlayStats = {
  bbox: {
    x: number
    y: number
    width: number
    height: number
  }
  maxValue: number
  actualAverage: number
  weightedAverage: number
  nonZeroPixels: number
  overlayFps: number
  overlayFrameCount: number
}

/** Matches processor_cli.py PROGRESS_PREFIX / emit_progress JSON lines. */
const PROGRESS_JSON_PREFIX = 'PROGRESS_JSON:'

type ProcessingProgress = {
  phase: string
  current: number
  total: number
  startedAt: string
  updatedAt: string
}

type SessionRecord = {
  id: string
  title: string
  youtubeUrl: string
  videoId: string
  createdAt: string
  updatedAt: string
  status: SessionStatus
  bbox: BBox | null
  fieldQuad: FieldQuad | null
  video: {
    fileName: string | null
    width: number | null
    height: number | null
    duration: number | null
  }
  overlay: {
    fileName: string
    transparentFileName: string
    framesDirName: string | null
    rawDataFileName: string
    fieldMapDataFileName: string | null
    stats: OverlayStats
  } | null
  lastError: string | null
  processingProgress: ProcessingProgress | null
}

const ROOT_DIR = process.cwd()
const IS_WIN = process.platform === 'win32'

/**
 * GUI/IDE-launched Bun often inherits a shorter PATH than an interactive terminal, so `ffmpeg.exe`
 * is ENOENT even when `where ffmpeg` works in PowerShell. We merge common install locations, then
 * resolve a full path with `where.exe`.
 */
function envWithWindowsPathExtras(): NodeJS.ProcessEnv {
  if (!IS_WIN) {
    return process.env
  }
  const pathKey = 'Path'
  const pf = process.env.ProgramFiles ?? 'C:\\Program Files'
  const pfx86 = process.env['ProgramFiles(x86)'] ?? ''
  const local = process.env.LOCALAPPDATA ?? ''
  const userProfile = process.env.USERPROFILE ?? ''
  const choco = process.env.ChocolateyInstall
  const extras = [
    path.join(pf, 'ffmpeg', 'bin'),
    path.join(pfx86, 'ffmpeg', 'bin'),
    'C:\\ffmpeg\\bin',
    path.join(local, 'Microsoft', 'WinGet', 'Links'),
    userProfile ? path.join(userProfile, 'scoop', 'shims') : '',
    choco ? path.join(choco, 'bin') : '',
  ].filter((p): p is string => Boolean(p && p.length > 2))
  const cur = process.env[pathKey] ?? ''
  const merged = [cur, ...extras].filter(Boolean).join(path.delimiter)
  return { ...process.env, [pathKey]: merged }
}

function resolveWindowsExecutable(exeName: string): string | null {
  if (!IS_WIN) {
    return null
  }
  try {
    const out = execSync(`where.exe ${exeName}`, {
      encoding: 'utf8',
      windowsHide: true,
      env: envWithWindowsPathExtras(),
    })
    const first = out
      .split(/\r?\n/)
      .map((l) => l.trim())
      .find((l) => l.length > 0 && !l.toLowerCase().startsWith('info:'))
    if (!first) {
      return null
    }
    return existsSync(first) ? first : null
  } catch {
    return null
  }
}

const SPAWN_ENV = IS_WIN ? envWithWindowsPathExtras() : process.env

const PYTHON_BIN = Bun.env.PYTHON_BIN ?? 'python'
const FFMPEG_BIN =
  Bun.env.FFMPEG_BIN ??
  (IS_WIN ? resolveWindowsExecutable('ffmpeg.exe') : null) ??
  (IS_WIN ? 'ffmpeg.exe' : 'ffmpeg')
const FFPROBE_BIN =
  Bun.env.FFPROBE_BIN ??
  (IS_WIN ? resolveWindowsExecutable('ffprobe.exe') : null) ??
  (IS_WIN ? 'ffprobe.exe' : 'ffprobe')
const PORT = Number(Bun.env.PORT ?? '3001')
const SESSIONS_DIR = path.join(ROOT_DIR, 'sessions')
const CLIENT_DIST_DIR = path.join(ROOT_DIR, 'webui', 'dist')

await mkdir(SESSIONS_DIR, { recursive: true })

function clamp(value: number, min = 0, max = 1) {
  return Math.min(Math.max(value, min), max)
}

function slugify(value: string) {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 48) || 'session'
}

function createSessionId(title: string, videoId: string) {
  return `${slugify(title)}-${videoId.toLowerCase().slice(0, 12)}`
}

function sessionDir(sessionId: string) {
  return path.join(SESSIONS_DIR, sessionId)
}

function sessionFile(sessionId: string) {
  return path.join(sessionDir(sessionId), 'session.json')
}

function sanitizeName(value: string) {
  if (!/^[a-z0-9._-]+$/i.test(value)) {
    throw new Error('Invalid file name')
  }
  return value
}

function json(data: unknown, init: ResponseInit = {}) {
  return Response.json(data, init)
}

function normalizeBBox(input: unknown) {
  if (!input || typeof input !== 'object') {
    return null
  }

  const raw = input as Record<string, unknown>
  const x = Number(raw.x)
  const y = Number(raw.y)
  const width = Number(raw.width)
  const height = Number(raw.height)

  if (![x, y, width, height].every(Number.isFinite)) {
    return null
  }

  if (width <= 0 || height <= 0) {
    return null
  }

  const normalizedX = clamp(x)
  const normalizedY = clamp(y)
  const normalizedWidth = clamp(width, 0.01, 1 - normalizedX)
  const normalizedHeight = clamp(height, 0.01, 1 - normalizedY)

  return {
    x: normalizedX,
    y: normalizedY,
    width: normalizedWidth,
    height: normalizedHeight,
  }
}

function denormalizeBBox(box: BBox, width: number, height: number) {
  const x = Math.round(box.x * width)
  const y = Math.round(box.y * height)
  const boxWidth = Math.max(1, Math.round(box.width * width))
  const boxHeight = Math.max(1, Math.round(box.height * height))
  return { x, y, width: boxWidth, height: boxHeight }
}

function normalizePoint(input: unknown) {
  if (!input || typeof input !== 'object') {
    return null
  }

  const raw = input as Record<string, unknown>
  const x = Number(raw.x)
  const y = Number(raw.y)
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    return null
  }

  return {
    x: clamp(x),
    y: clamp(y),
  }
}

/** [TL, TR, BR, BL] for cv2.getPerspectiveTransform — same heuristic as App.tsx. */
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

function normalizeFieldQuad(input: unknown) {
  if (!Array.isArray(input) || input.length !== 4) {
    return null
  }

  const points = input.map(normalizePoint)
  if (points.some((point) => !point)) {
    return null
  }

  return orderQuadPoints(points as Point[])
}

function bboxFromQuad(quad: FieldQuad, padding = 0.01) {
  const xs = quad.map((point) => point.x)
  const ys = quad.map((point) => point.y)
  const minX = clamp(Math.min(...xs) - padding)
  const minY = clamp(Math.min(...ys) - padding)
  const maxX = clamp(Math.max(...xs) + padding)
  const maxY = clamp(Math.max(...ys) + padding)

  return {
    x: minX,
    y: minY,
    width: Math.max(0.01, maxX - minX),
    height: Math.max(0.01, maxY - minY),
  }
}

function denormalizeFieldQuad(quad: FieldQuad, width: number, height: number) {
  return quad.map((point) => ({
    x: Math.round(point.x * width),
    y: Math.round(point.y * height),
  }))
}

async function readSessionRecord(sessionId: string) {
  const content = await readFile(sessionFile(sessionId), 'utf8')
  const parsed = JSON.parse(content) as SessionRecord
  return {
    ...parsed,
    fieldQuad: parsed.fieldQuad ?? null,
    processingProgress: parsed.processingProgress ?? null,
  }
}

async function writeSessionRecord(record: SessionRecord) {
  await mkdir(sessionDir(record.id), { recursive: true })
  await writeFile(sessionFile(record.id), `${JSON.stringify(record, null, 2)}\n`, 'utf8')
}

function toClientSession(record: SessionRecord) {
  const overlayFramesDirName =
    record.overlay?.framesDirName && existsSync(path.join(sessionDir(record.id), record.overlay.framesDirName))
      ? record.overlay.framesDirName
      : null
  const fieldMapDataFileName =
    record.overlay?.fieldMapDataFileName && existsSync(path.join(sessionDir(record.id), record.overlay.fieldMapDataFileName))
      ? record.overlay.fieldMapDataFileName
      : existsSync(path.join(sessionDir(record.id), 'field-map.json'))
        ? 'field-map.json'
        : null

  return {
    ...record,
    media: {
      videoUrl:
        record.video.fileName ? `/media/${record.id}/${encodeURIComponent(record.video.fileName)}` : null,
      overlayUrl:
        record.overlay?.fileName ? `/media/${record.id}/${encodeURIComponent(record.overlay.fileName)}` : null,
      overlayTransparentUrl:
        record.overlay?.transparentFileName
          ? `/media/${record.id}/${encodeURIComponent(record.overlay.transparentFileName)}`
          : null,
      overlayFrameUrlTemplate:
        overlayFramesDirName
          ? `/media-frame/${record.id}/__FRAME__.webp`
          : null,
      fieldMapDataUrl:
        fieldMapDataFileName
          ? `/media/${record.id}/${encodeURIComponent(fieldMapDataFileName)}`
          : null,
    },
  }
}

async function listSessions() {
  const entries = await readdir(SESSIONS_DIR, { withFileTypes: true })
  const sessions: SessionRecord[] = []

  for (const entry of entries) {
    if (!entry.isDirectory()) {
      continue
    }

    const filePath = sessionFile(entry.name)
    if (!existsSync(filePath)) {
      continue
    }

    sessions.push(await readSessionRecord(entry.name))
  }

  sessions.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))
  return sessions
}

function spawnNotFoundMessage(command: string, err: NodeJS.ErrnoException) {
  const base = `Could not start "${command}" (${err.message}).`
  if (/ffmpeg|ffprobe/i.test(command)) {
    return `${base} Install FFmpeg, then set FFMPEG_BIN and FFPROBE_BIN to the full paths from PowerShell, e.g. (Get-Command ffmpeg).Source. The IDE often uses a shorter PATH than your terminal; absolute paths avoid that.`
  }
  return `${base} Check that the program is installed and on PATH, or set the corresponding *_BIN environment variable.`
}

async function runCommand(command: string, args: string[], cwd = ROOT_DIR) {
  return await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn(command, args, { cwd, windowsHide: true, env: SPAWN_ENV })
    let stdout = ''
    let stderr = ''

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString()
    })

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString()
    })

    child.on('error', (err) => {
      const e = err as NodeJS.ErrnoException
      if (e.code === 'ENOENT') {
        reject(new Error(spawnNotFoundMessage(command, e)))
        return
      }
      reject(err)
    })
    child.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout, stderr })
        return
      }

      reject(new Error(stderr.trim() || stdout.trim() || `${command} exited with code ${code}`))
    })
  })
}

async function runProcessorWithProgress(
  record: SessionRecord,
  args: string[],
  runStartedAt: string,
): Promise<{ stdout: string; stderr: string }> {
  return await new Promise((resolve, reject) => {
    const child = spawn(PYTHON_BIN, ['-u', ...args], {
      cwd: ROOT_DIR,
      windowsHide: true,
      env: { ...SPAWN_ENV, PYTHONUNBUFFERED: '1' },
    })
    let stdout = ''
    let stderr = ''
    let stderrLineBuf = ''

    let writeTimer: ReturnType<typeof setTimeout> | null = null
    const flushRecord = () => {
      void writeSessionRecord(record).catch(() => {})
    }
    const schedulePersist = () => {
      if (writeTimer) {
        clearTimeout(writeTimer)
      }
      writeTimer = setTimeout(() => {
        writeTimer = null
        flushRecord()
      }, 250)
    }

    const applyProgressLine = (jsonPart: string) => {
      try {
        const payload = JSON.parse(jsonPart) as { phase?: string; current?: number; total?: number }
        if (!payload.phase || typeof payload.current !== 'number' || typeof payload.total !== 'number') {
          return
        }
        const now = new Date().toISOString()
        record.processingProgress = {
          phase: payload.phase,
          current: payload.current,
          total: Math.max(1, payload.total),
          startedAt: runStartedAt,
          updatedAt: now,
        }
        record.updatedAt = now
        schedulePersist()
      } catch {
        /* ignore malformed progress lines */
      }
    }

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString()
    })

    child.stderr.on('data', (chunk) => {
      const text = chunk.toString()
      stderr += text
      stderrLineBuf += text
      const lines = stderrLineBuf.split('\n')
      stderrLineBuf = lines.pop() ?? ''
      for (const line of lines) {
        if (line.startsWith(PROGRESS_JSON_PREFIX)) {
          applyProgressLine(line.slice(PROGRESS_JSON_PREFIX.length))
        }
      }
    })

    child.on('error', reject)
    child.on('close', (code) => {
      if (writeTimer) {
        clearTimeout(writeTimer)
        writeTimer = null
      }
      if (stderrLineBuf.startsWith(PROGRESS_JSON_PREFIX)) {
        applyProgressLine(stderrLineBuf.slice(PROGRESS_JSON_PREFIX.length))
      }
      flushRecord()
      if (code === 0) {
        resolve({ stdout, stderr })
        return
      }
      reject(new Error(stderr.trim() || stdout.trim() || `${PYTHON_BIN} exited with code ${code}`))
    })
  })
}

async function getYouTubeMetadata(url: string) {
  const { stdout } = await runCommand(PYTHON_BIN, [
    '-m',
    'yt_dlp',
    '--dump-single-json',
    '--skip-download',
    '--no-playlist',
    '--no-warnings',
    url,
  ])

  const parsed = JSON.parse(stdout)
  return {
    title: parsed.title as string,
    videoId: parsed.id as string,
    width: typeof parsed.width === 'number' ? parsed.width : null,
    height: typeof parsed.height === 'number' ? parsed.height : null,
    duration: typeof parsed.duration === 'number' ? parsed.duration : null,
  }
}

async function findDownloadedVideoFile(sessionId: string) {
  const entries = await readdir(sessionDir(sessionId))
  const fileName = entries.find((entry) => /^video\.(mp4|mkv|mov|webm)$/i.test(entry))
  return fileName ?? null
}

async function downloadVideo(url: string, record: SessionRecord) {
  await runCommand(PYTHON_BIN, [
    '-m',
    'yt_dlp',
    '--no-playlist',
    '-f',
    'best[ext=mp4][acodec!=none][vcodec!=none]/best[ext=mp4]/best',
    '-o',
    path.join(sessionDir(record.id), 'video.%(ext)s'),
    url,
  ])

  const fileName = await findDownloadedVideoFile(record.id)
  if (!fileName) {
    throw new Error('The download completed but no local video file was found.')
  }

  record.video.fileName = fileName
}

async function probeVideoDurationSeconds(filePath: string): Promise<number> {
  const { stdout } = await runCommand(FFPROBE_BIN, [
    '-v',
    'error',
    '-show_entries',
    'format=duration',
    '-of',
    'default=noprint_wrappers=1:nokey=1',
    filePath,
  ])
  const value = Number.parseFloat(stdout.trim())
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error('Could not read video duration.')
  }
  return value
}

async function trimSessionVideo(sessionId: string, trimStartSec: number, trimEndSec: number) {
  const record = await readSessionRecord(sessionId)
  if (!record.video.fileName) {
    throw new Error('No video file for this session.')
  }

  const dir = sessionDir(sessionId)
  const inputPath = path.join(dir, record.video.fileName)
  if (!existsSync(inputPath)) {
    throw new Error('Video file is missing on disk.')
  }

  const metaDuration = record.video.duration ?? (await probeVideoDurationSeconds(inputPath))
  let start = Math.max(0, trimStartSec)
  let end = Math.min(metaDuration, trimEndSec)
  if (end <= start) {
    throw new Error('End time must be after start time.')
  }
  const length = end - start

  const ext = path.extname(record.video.fileName) || '.mp4'
  const tmpOut = path.join(dir, `video-trimmed${ext}`)

  try {
    await runCommand(FFMPEG_BIN, [
      '-y',
      '-ss',
      String(start),
      '-i',
      inputPath,
      '-t',
      String(length),
      '-c',
      'copy',
      '-movflags',
      '+faststart',
      tmpOut,
    ])
  } catch {
    await runCommand(FFMPEG_BIN, [
      '-y',
      '-ss',
      String(start),
      '-i',
      inputPath,
      '-t',
      String(length),
      '-c:v',
      'libx264',
      '-preset',
      'fast',
      '-crf',
      '20',
      '-c:a',
      'aac',
      '-b:a',
      '192k',
      '-movflags',
      '+faststart',
      tmpOut,
    ])
  }

  const newDuration = await probeVideoDurationSeconds(tmpOut)
  await rm(inputPath)
  await rename(tmpOut, path.join(dir, record.video.fileName))

  record.video.duration = newDuration
  record.updatedAt = new Date().toISOString()
  await writeSessionRecord(record)
  return record
}

async function importSession(url: string): Promise<{ record: SessionRecord; videoJustDownloaded: boolean }> {
  let videoJustDownloaded = false
  const metadata = await getYouTubeMetadata(url)
  const id = createSessionId(metadata.title, metadata.videoId)
  const now = new Date().toISOString()
  const existingPath = sessionFile(id)

  let record: SessionRecord
  if (existsSync(existingPath)) {
    record = await readSessionRecord(id)
    record.youtubeUrl = url
    record.title = metadata.title
    record.fieldQuad = record.fieldQuad ?? null
    record.video.width = metadata.width
    record.video.height = metadata.height
    record.video.duration = metadata.duration
    if (record.video.fileName && existsSync(path.join(sessionDir(id), record.video.fileName))) {
      record.status = record.overlay ? 'completed' : 'ready'
    }
    record.updatedAt = now
  } else {
    record = {
      id,
      title: metadata.title,
      youtubeUrl: url,
      videoId: metadata.videoId,
      createdAt: now,
      updatedAt: now,
      status: 'downloading',
      bbox: null,
      fieldQuad: null,
      video: {
        fileName: null,
        width: metadata.width,
        height: metadata.height,
        duration: metadata.duration,
      },
      overlay: null,
      lastError: null,
      processingProgress: null,
    }
  }

  await writeSessionRecord(record)

  if (!record.video.fileName || !existsSync(path.join(sessionDir(id), record.video.fileName))) {
    record.status = 'downloading'
    record.lastError = null
    record.updatedAt = new Date().toISOString()
    await writeSessionRecord(record)

    try {
      await downloadVideo(url, record)
      videoJustDownloaded = true
      record.status = record.overlay ? 'completed' : 'ready'
      record.updatedAt = new Date().toISOString()
      await writeSessionRecord(record)
    } catch (error) {
      record.status = 'error'
      record.lastError = error instanceof Error ? error.message : 'Video download failed.'
      record.updatedAt = new Date().toISOString()
      await writeSessionRecord(record)
      throw error
    }
  }

  return { record, videoJustDownloaded }
}

async function updateSession(sessionId: string, mutator: (record: SessionRecord) => void | Promise<void>) {
  const record = await readSessionRecord(sessionId)
  await mutator(record)
  record.updatedAt = new Date().toISOString()
  await writeSessionRecord(record)
  return record
}

async function processSession(sessionId: string) {
  const record = await readSessionRecord(sessionId)

  if (!record.video.fileName) {
    throw new Error('The session does not have a downloaded video yet.')
  }

  const normalizedBBox = record.fieldQuad ? bboxFromQuad(record.fieldQuad) : record.bbox
  if (!normalizedBBox) {
    throw new Error('Mark the field borders before running the analysis.')
  }

  if (!record.video.width || !record.video.height) {
    throw new Error('Video dimensions are missing. Load the video once and try again.')
  }

  const pixelBox = denormalizeBBox(normalizedBBox, record.video.width, record.video.height)
  const pixelQuad = record.fieldQuad
    ? denormalizeFieldQuad(record.fieldQuad, record.video.width, record.video.height)
    : null
  const videoPath = path.join(sessionDir(sessionId), record.video.fileName)

  const runStartedAt = new Date().toISOString()
  record.status = 'processing'
  record.lastError = null
  record.processingProgress = {
    phase: 'starting',
    current: 0,
    total: 1,
    startedAt: runStartedAt,
    updatedAt: runStartedAt,
  }
  record.updatedAt = runStartedAt
  await writeSessionRecord(record)

  const logPath = path.join(sessionDir(sessionId), 'process.log')
  try {
    const { stdout, stderr } = await runProcessorWithProgress(
      record,
      [
        'processor_cli.py',
        '--video',
        videoPath,
        '--session-dir',
        sessionDir(sessionId),
        '--bbox',
        `${pixelBox.x},${pixelBox.y},${pixelBox.width},${pixelBox.height}`,
        ...(pixelQuad
          ? [
              '--quad',
              pixelQuad.map((point) => `${point.x},${point.y}`).join(','),
            ]
          : []),
      ],
      runStartedAt,
    )
    await writeFile(
      logPath,
      [stdout, stderr].filter(Boolean).join(stdout && stderr ? '\n--- stderr ---\n' : '\n'),
      'utf8',
    )

    const statsPath = path.join(sessionDir(sessionId), 'stats.json')
    const stats = JSON.parse(await readFile(statsPath, 'utf8')) as OverlayStats

    record.overlay = {
      fileName: 'overlay.png',
      transparentFileName: 'overlay-transparent.png',
      framesDirName: 'overlay-frames',
      rawDataFileName: 'raw_data.txt',
      fieldMapDataFileName: 'field-map.json',
      stats,
    }
    record.bbox = normalizedBBox
    record.status = 'completed'
    record.processingProgress = null
    record.updatedAt = new Date().toISOString()
    await writeSessionRecord(record)
    return record
  } catch (error) {
    record.status = 'error'
    record.lastError = error instanceof Error ? error.message : 'Overlay generation failed.'
    record.processingProgress = null
    record.updatedAt = new Date().toISOString()
    await writeSessionRecord(record)
    try {
      await writeFile(logPath, record.lastError, 'utf8')
    } catch {
      /* ignore log write errors */
    }
    throw error
  }
}

async function parseBody<T>(request: Request) {
  try {
    return (await request.json()) as T
  } catch {
    return null
  }
}

function inferMimeType(fileName: string) {
  const extension = path.extname(fileName).toLowerCase()
  if (extension === '.png') {
    return 'image/png'
  }
  if (extension === '.jpg' || extension === '.jpeg') {
    return 'image/jpeg'
  }
  if (extension === '.mp4') {
    return 'video/mp4'
  }
  if (extension === '.webm') {
    return 'video/webm'
  }
  if (extension === '.txt') {
    return 'text/plain; charset=utf-8'
  }
  if (extension === '.json') {
    return 'application/json; charset=utf-8'
  }
  return 'application/octet-stream'
}

function cacheControlForSessionFile(filePath: string) {
  const base = path.basename(filePath).toLowerCase()
  if (base.endsWith('.json') || base.endsWith('.txt')) {
    return 'no-store'
  }
  return 'public, max-age=3600'
}

function createRangeResponse(filePath: string, request: Request, contentType: string) {
  const file = Bun.file(filePath)
  const size = file.size
  const rangeHeader = request.headers.get('range')
  const cacheControl = cacheControlForSessionFile(filePath)

  if (!rangeHeader || !Number.isFinite(size) || size <= 0) {
    return new Response(file, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(size),
        'Accept-Ranges': 'bytes',
        'Cache-Control': cacheControl,
      },
    })
  }

  const match = /^bytes=(\d*)-(\d*)$/.exec(rangeHeader.trim())
  if (!match) {
    return new Response(null, {
      status: 416,
      headers: {
        'Content-Range': `bytes */${size}`,
      },
    })
  }

  const startToken = match[1]
  const endToken = match[2]

  let start = startToken ? Number(startToken) : NaN
  let end = endToken ? Number(endToken) : NaN

  if (Number.isNaN(start)) {
    const suffixLength = Number.isNaN(end) ? size : end
    start = Math.max(size - suffixLength, 0)
    end = size - 1
  } else {
    if (Number.isNaN(end) || end >= size) {
      end = size - 1
    }
  }

  if (start < 0 || start >= size || end < start) {
    return new Response(null, {
      status: 416,
      headers: {
        'Content-Range': `bytes */${size}`,
      },
    })
  }

  return new Response(file.slice(start, end + 1), {
    status: 206,
    headers: {
      'Content-Type': contentType,
      'Content-Length': String(end - start + 1),
      'Content-Range': `bytes ${start}-${end}/${size}`,
      'Accept-Ranges': 'bytes',
      'Cache-Control': cacheControl,
    },
  })
}

async function serveClientAsset(urlPath: string) {
  const normalizedPath = urlPath === '/' ? 'index.html' : urlPath.replace(/^\/+/, '')
  const resolved = path.join(CLIENT_DIST_DIR, normalizedPath)

  if (existsSync(resolved) && !resolved.endsWith(path.sep)) {
    return new Response(Bun.file(resolved))
  }

  const indexPath = path.join(CLIENT_DIST_DIR, 'index.html')
  if (existsSync(indexPath)) {
    return new Response(Bun.file(indexPath), {
      headers: {
        'Content-Type': 'text/html; charset=utf-8',
      },
    })
  }

  return json(
    {
      error: 'Frontend build not found. Run "bun run build" or use "bun run dev" during development.',
    },
    { status: 503 },
  )
}

Bun.serve({
  port: PORT,
  async fetch(request) {
    const url = new URL(request.url)

    try {
      if (request.method === 'GET' && url.pathname === '/api/health') {
        return json({
          ok: true,
          sessionsDir: SESSIONS_DIR,
          pythonBin: PYTHON_BIN,
          ffmpegBin: FFMPEG_BIN,
          ffprobeBin: FFPROBE_BIN,
        })
      }

      if (request.method === 'GET' && url.pathname === '/api/sessions') {
        const sessions = await listSessions()
        return json(sessions.map(toClientSession))
      }

      if (request.method === 'POST' && url.pathname === '/api/sessions/import') {
        const body = await parseBody<{ url?: string }>(request)
        const youtubeUrl = body?.url?.trim()

        if (!youtubeUrl) {
          return json({ error: 'Paste a YouTube link first.' }, { status: 400 })
        }

        const { record, videoJustDownloaded } = await importSession(youtubeUrl)
        return json({ session: toClientSession(record), videoJustDownloaded })
      }

      const trimMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)\/trim-video$/)
      if (request.method === 'POST' && trimMatch) {
        const body = await parseBody<{ trimStartSec?: number; trimEndSec?: number }>(request)
        const start = Number(body?.trimStartSec)
        const end = Number(body?.trimEndSec)
        if (!Number.isFinite(start) || !Number.isFinite(end)) {
          return json({ error: 'trimStartSec and trimEndSec must be numbers (seconds).' }, { status: 400 })
        }
        try {
          const record = await trimSessionVideo(trimMatch[1], start, end)
          return json(toClientSession(record))
        } catch (error) {
          return json(
            { error: error instanceof Error ? error.message : 'Could not trim video.' },
            { status: 400 },
          )
        }
      }

      const sessionMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)$/)
      if (request.method === 'GET' && sessionMatch) {
        const record = await readSessionRecord(sessionMatch[1])
        return json(toClientSession(record))
      }

      const bboxMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)\/bbox$/)
      if (request.method === 'POST' && bboxMatch) {
        const body = await parseBody<{ bbox?: BBox | null }>(request)
        const normalizedBBox = body?.bbox ? normalizeBBox(body.bbox) : null

        const record = await updateSession(bboxMatch[1], (session) => {
          session.bbox = normalizedBBox
          session.fieldQuad = null
          session.overlay = null
          session.status = normalizedBBox ? 'ready' : 'idle'
        })

        return json(toClientSession(record))
      }

      const fieldQuadMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)\/field-quad$/)
      if (request.method === 'POST' && fieldQuadMatch) {
        const body = await parseBody<{ fieldQuad?: FieldQuad | null }>(request)
        if (!body || typeof body !== 'object' || !('fieldQuad' in body)) {
          return json({ error: 'Request body must include fieldQuad (or null to clear).' }, { status: 400 })
        }

        let normalizedFieldQuad: FieldQuad | null
        if (body.fieldQuad === null) {
          normalizedFieldQuad = null
        } else {
          const normalized = normalizeFieldQuad(body.fieldQuad)
          if (!normalized) {
            return json(
              {
                error:
                  'Could not save that field outline. Use four corners in the video frame, or pick slightly wider corners if the shape is too thin.',
              },
              { status: 400 },
            )
          }
          normalizedFieldQuad = normalized
        }

        const record = await updateSession(fieldQuadMatch[1], (session) => {
          session.fieldQuad = normalizedFieldQuad
          session.bbox = normalizedFieldQuad ? bboxFromQuad(normalizedFieldQuad) : null
          session.overlay = null
          session.status = normalizedFieldQuad ? 'ready' : 'idle'
        })

        return json(toClientSession(record))
      }

      const metadataMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)\/video-metadata$/)
      if (request.method === 'POST' && metadataMatch) {
        const body = await parseBody<{ width?: number; height?: number; duration?: number }>(request)
        const record = await updateSession(metadataMatch[1], (session) => {
          session.video.width = typeof body?.width === 'number' ? body.width : session.video.width
          session.video.height = typeof body?.height === 'number' ? body.height : session.video.height
          session.video.duration = typeof body?.duration === 'number' ? body.duration : session.video.duration
        })

        return json(toClientSession(record))
      }

      const processMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)\/process$/)
      if (request.method === 'POST' && processMatch) {
        const record = await processSession(processMatch[1])
        return json(toClientSession(record))
      }

      const processLogMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)\/process-log$/)
      if (request.method === 'GET' && processLogMatch) {
        const sid = processLogMatch[1]
        const logPath = path.join(sessionDir(sid), 'process.log')
        let text = ''
        if (existsSync(logPath)) {
          text = await readFile(logPath, 'utf8')
        } else {
          const record = await readSessionRecord(sid)
          if (record.lastError) {
            text = `Last error: ${record.lastError}`
          }
        }
        return new Response(text, {
          headers: { 'Content-Type': 'text/plain; charset=utf-8' },
        })
      }

      const deleteMatch = url.pathname.match(/^\/api\/sessions\/([^/]+)$/)
      if (request.method === 'DELETE' && deleteMatch) {
        await rm(sessionDir(deleteMatch[1]), { recursive: true, force: true })
        return json({ ok: true })
      }

      const mediaMatch = url.pathname.match(/^\/media\/([^/]+)\/([^/]+)$/)
      if (request.method === 'GET' && mediaMatch) {
        const sessionId = mediaMatch[1]
        const fileName = sanitizeName(decodeURIComponent(mediaMatch[2]))
        const filePath = path.join(sessionDir(sessionId), fileName)

        if (!existsSync(filePath)) {
          return json({ error: 'File not found.' }, { status: 404 })
        }

        return createRangeResponse(filePath, request, inferMimeType(fileName))
      }

      const frameMatch = url.pathname.match(/^\/media-frame\/([^/]+)\/(\d+)\.webp$/)
      if (request.method === 'GET' && frameMatch) {
        const sessionId = frameMatch[1]
        const frameIndex = Number(frameMatch[2])
        const filePath = path.join(sessionDir(sessionId), 'overlay-frames', `frame_${frameIndex.toString().padStart(6, '0')}.webp`)

        if (!existsSync(filePath)) {
          return json({ error: 'Frame not found.' }, { status: 404 })
        }

        return new Response(Bun.file(filePath), {
          headers: {
            'Content-Type': 'image/webp',
            'Cache-Control': 'private, max-age=0, must-revalidate',
          },
        })
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unexpected server error.'
      return json({ error: message }, { status: 500 })
    }

    return serveClientAsset(url.pathname)
  },
})

console.log(`Fuel density map server listening on http://localhost:${PORT}`)
