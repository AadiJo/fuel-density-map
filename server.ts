import { spawn } from 'node:child_process'
import { existsSync } from 'node:fs'
import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'

type SessionStatus = 'idle' | 'downloading' | 'ready' | 'processing' | 'completed' | 'error'

type BBox = {
  x: number
  y: number
  width: number
  height: number
}

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

type SessionRecord = {
  id: string
  title: string
  youtubeUrl: string
  videoId: string
  createdAt: string
  updatedAt: string
  status: SessionStatus
  bbox: BBox | null
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
    stats: OverlayStats
  } | null
  lastError: string | null
}

const ROOT_DIR = process.cwd()
const PYTHON_BIN = Bun.env.PYTHON_BIN ?? 'python'
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

async function readSessionRecord(sessionId: string) {
  const content = await readFile(sessionFile(sessionId), 'utf8')
  return JSON.parse(content) as SessionRecord
}

async function writeSessionRecord(record: SessionRecord) {
  await mkdir(sessionDir(record.id), { recursive: true })
  await writeFile(sessionFile(record.id), `${JSON.stringify(record, null, 2)}\n`, 'utf8')
}

function toClientSession(record: SessionRecord) {
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
        record.overlay?.framesDirName
          ? `/media-frame/${record.id}/__FRAME__.webp`
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

async function runCommand(command: string, args: string[], cwd = ROOT_DIR) {
  return await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
    const child = spawn(command, args, { cwd, windowsHide: true })
    let stdout = ''
    let stderr = ''

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString()
    })

    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString()
    })

    child.on('error', reject)
    child.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout, stderr })
        return
      }

      reject(new Error(stderr.trim() || stdout.trim() || `${command} exited with code ${code}`))
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

async function importSession(url: string) {
  const metadata = await getYouTubeMetadata(url)
  const id = createSessionId(metadata.title, metadata.videoId)
  const now = new Date().toISOString()
  const existingPath = sessionFile(id)

  let record: SessionRecord
  if (existsSync(existingPath)) {
    record = await readSessionRecord(id)
    record.youtubeUrl = url
    record.title = metadata.title
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
      video: {
        fileName: null,
        width: metadata.width,
        height: metadata.height,
        duration: metadata.duration,
      },
      overlay: null,
      lastError: null,
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

  return record
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

  if (!record.bbox) {
    throw new Error('Draw a bounding box before running the analysis.')
  }

  if (!record.video.width || !record.video.height) {
    throw new Error('Video dimensions are missing. Load the video once and try again.')
  }

  const pixelBox = denormalizeBBox(record.bbox, record.video.width, record.video.height)
  const videoPath = path.join(sessionDir(sessionId), record.video.fileName)

  record.status = 'processing'
  record.lastError = null
  record.updatedAt = new Date().toISOString()
  await writeSessionRecord(record)

  try {
    await runCommand(PYTHON_BIN, [
      'processor_cli.py',
      '--video',
      videoPath,
      '--session-dir',
      sessionDir(sessionId),
      '--bbox',
      `${pixelBox.x},${pixelBox.y},${pixelBox.width},${pixelBox.height}`,
    ])

    const statsPath = path.join(sessionDir(sessionId), 'stats.json')
    const stats = JSON.parse(await readFile(statsPath, 'utf8')) as OverlayStats

    record.overlay = {
      fileName: 'overlay.png',
      transparentFileName: 'overlay-transparent.png',
      framesDirName: 'overlay-frames',
      rawDataFileName: 'raw_data.txt',
      stats,
    }
    record.status = 'completed'
    record.updatedAt = new Date().toISOString()
    await writeSessionRecord(record)
    return record
  } catch (error) {
    record.status = 'error'
    record.lastError = error instanceof Error ? error.message : 'Overlay generation failed.'
    record.updatedAt = new Date().toISOString()
    await writeSessionRecord(record)
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

function createRangeResponse(filePath: string, request: Request, contentType: string) {
  const file = Bun.file(filePath)
  const size = file.size
  const rangeHeader = request.headers.get('range')

  if (!rangeHeader || !Number.isFinite(size) || size <= 0) {
    return new Response(file, {
      headers: {
        'Content-Type': contentType,
        'Content-Length': String(size),
        'Accept-Ranges': 'bytes',
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

        const record = await importSession(youtubeUrl)
        return json(toClientSession(record))
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
          session.overlay = null
          session.status = normalizedBBox ? 'ready' : 'idle'
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
            'Cache-Control': 'public, max-age=31536000, immutable',
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
