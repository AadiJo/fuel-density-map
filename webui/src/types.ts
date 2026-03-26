export type SessionStatus = 'idle' | 'downloading' | 'ready' | 'processing' | 'completed' | 'error'

export type BBox = {
  x: number
  y: number
  width: number
  height: number
}

export type Point = {
  x: number
  y: number
}

export type FieldQuad = [Point, Point, Point, Point]

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

export type FieldMapPoint = [number, number, number]

export type FieldMapData = {
  imageWidth: number
  imageHeight: number
  fps: number
  frameCount: number
  frames: FieldMapPoint[][]
}

export type ProcessingProgress = {
  phase: string
  current: number
  total: number
  startedAt: string
  updatedAt: string
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
  media: {
    videoUrl: string | null
    overlayUrl: string | null
    overlayTransparentUrl: string | null
    overlayFrameUrlTemplate: string | null
    fieldMapDataUrl: string | null
  }
  lastError: string | null
  processingProgress: ProcessingProgress | null
}

export type DisplayMode = 'match' | 'field'
