import type { BBox, Session } from './types'

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
  listSessions() {
    return request<Session[]>('/api/sessions')
  },
  importSession(url: string) {
    return request<Session>('/api/sessions/import', {
      method: 'POST',
      body: JSON.stringify({ url }),
    })
  },
  saveBBox(sessionId: string, bbox: BBox | null) {
    return request<Session>(`/api/sessions/${sessionId}/bbox`, {
      method: 'POST',
      body: JSON.stringify({ bbox }),
    })
  },
  updateVideoMetadata(sessionId: string, width: number, height: number, duration: number) {
    return request<Session>(`/api/sessions/${sessionId}/video-metadata`, {
      method: 'POST',
      body: JSON.stringify({ width, height, duration }),
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
}
