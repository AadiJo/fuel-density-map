export type SessionStatus = 'idle' | 'downloading' | 'ready' | 'processing' | 'completed' | 'error'

export type BBox = {
  x: number
  y: number
  width: number
  height: number
}

export type OverlayStats = {
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

export type Session = {
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
  media: {
    videoUrl: string | null
    overlayUrl: string | null
    overlayTransparentUrl: string | null
    overlayFrameUrlTemplate: string | null
  }
  lastError: string | null
}

export type DisplayMode = 'video' | 'overlay' | 'blend'
