import type {
  AirProfileData,
  BBox,
  FieldMapData,
  FieldQuad,
  RGBColor,
  Session,
  StreamClipperOcrRegion,
  StreamClipperStatus,
  WallSide,
} from './types'

export type ImportSessionResult = {
  session: Session
  /** True when a new download just finished (not a cached session). */
  videoJustDownloaded: boolean
}

export type StartStreamClipperRequest = {
  url: string
  preRollSec: number
  postRollSec: number
  rollingBufferSec: number
  segmentTimeSec: number
  startThresholdSec: number
  maxMatchSec: number
  stopGraceSec: number
  minTimerConfidence: number
  ocrRegion: StreamClipperOcrRegion
}

async function request<T>(input: string, init?: RequestInit) {
  const response = await fetch(input, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
  })

  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(payload?.error ?? 'Request failed.')
  }

  return payload as T
}

export const api = {
  getSession(sessionId: string) {
    return request<Session>(`/api/sessions/${sessionId}`)
  },
  listSessions() {
    return request<Session[]>('/api/sessions')
  },
  importSession(url: string) {
    return request<ImportSessionResult>('/api/sessions/import', {
      method: 'POST',
      body: JSON.stringify({ url }),
    })
  },
  trimVideo(sessionId: string, trimStartSec: number, trimEndSec: number) {
    return request<Session>(`/api/sessions/${sessionId}/trim-video`, {
      method: 'POST',
      body: JSON.stringify({ trimStartSec, trimEndSec }),
    })
  },
  saveBBox(sessionId: string, bbox: BBox | null) {
    return request<Session>(`/api/sessions/${sessionId}/bbox`, {
      method: 'POST',
      body: JSON.stringify({ bbox }),
    })
  },
  saveFieldQuad(sessionId: string, fieldQuad: FieldQuad | null) {
    return request<Session>(`/api/sessions/${sessionId}/field-quad`, {
      method: 'POST',
      body: JSON.stringify({ fieldQuad }),
    })
  },
  saveWallQuad(sessionId: string, side: WallSide, wallQuad: FieldQuad | null) {
    return request<Session>(`/api/sessions/${sessionId}/wall-quads/${side}`, {
      method: 'POST',
      body: JSON.stringify({ wallQuad }),
    })
  },
  updateVideoMetadata(sessionId: string, width: number, height: number, duration: number) {
    return request<Session>(`/api/sessions/${sessionId}/video-metadata`, {
      method: 'POST',
      body: JSON.stringify({ width, height, duration }),
    })
  },
  saveFuelBaseColor(sessionId: string, fuelBaseColor: RGBColor) {
    return request<Session>(`/api/sessions/${sessionId}/fuel-base-color`, {
      method: 'POST',
      body: JSON.stringify({ fuelBaseColor }),
    })
  },
  sampleFuelBaseColor(sessionId: string, x: number, y: number, timeSec: number) {
    return request<Session>(`/api/sessions/${sessionId}/fuel-base-color/sample`, {
      method: 'POST',
      body: JSON.stringify({ x, y, timeSec }),
    })
  },
  processSession(sessionId: string) {
    return request<Session>(`/api/sessions/${sessionId}/process`, {
      method: 'POST',
      body: JSON.stringify({}),
    })
  },
  deleteSession(sessionId: string) {
    return request<{ ok: boolean }>(`/api/sessions/${sessionId}`, {
      method: 'DELETE',
    })
  },
  getFieldMapData(url: string) {
    return request<FieldMapData>(url, { cache: 'no-store' })
  },
  getAirProfileData(url: string) {
    return request<AirProfileData>(url, { cache: 'no-store' })
  },
  async getProcessLog(sessionId: string) {
    const response = await fetch(`/api/sessions/${sessionId}/process-log`, { cache: 'no-store' })
    if (!response.ok) {
      const payload = await response.json().catch(() => null)
      throw new Error(payload?.error ?? 'Request failed.')
    }
    return response.text()
  },
  getStreamClipperStatus() {
    return request<StreamClipperStatus>('/api/stream-clipper/status')
  },
  startStreamClipper(payload: StartStreamClipperRequest) {
    return request<StreamClipperStatus>('/api/stream-clipper/start', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
  },
  stopStreamClipper() {
    return request<StreamClipperStatus>('/api/stream-clipper/stop', {
      method: 'POST',
      body: JSON.stringify({}),
    })
  },
  saveStreamClipperOcrRegion(ocrRegion: StreamClipperOcrRegion) {
    return request<StreamClipperStatus>('/api/stream-clipper/ocr-region', {
      method: 'POST',
      body: JSON.stringify({ ocrRegion }),
    })
  },
}
