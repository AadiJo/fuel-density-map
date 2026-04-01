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
export type WallSide = 'left' | 'right'
export type WallQuads = Partial<Record<WallSide, FieldQuad>>

export type RGBColor = {
  r: number
  g: number
  b: number
}

export type OverlayStats = {
  backend?: 'cpu' | 'cuda'
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
  timings?: Record<string, number>
  rawCenterCountSummary?: {
    min: number
    p50: number
    p90: number
    p95: number
    max: number
    mean: number
  }
  stableTrackCountSummary?: {
    min: number
    p50: number
    p90: number
    p95: number
    max: number
    mean: number
  }
  detectorBudgetHits?: number
  saturatedFrameCount?: number
}

export type FieldMapPoint = [number, number, number]

export type FieldMapData = {
  imageWidth: number
  imageHeight: number
  fps: number
  frameCount: number
  frames: FieldMapPoint[][]
}

export type AirProfilePoint = [number, number]

export type AirProfileData = {
  fps: number
  frameCount: number
  wallSide?: 'top' | 'bottom' | 'left' | 'right' | 'mixed'
  frames: AirProfilePoint[][]
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
  fuelBaseColor: RGBColor
  bbox: BBox | null
  fieldQuad: FieldQuad | null
  wallQuad: FieldQuad | null
  wallQuads: WallQuads
  video: {
    fileName: string | null
    width: number | null
    height: number | null
    duration: number | null
  }
  overlay: {
    fileName: string
    transparentFileName: string
    overlayVideoFileName: string | null
    playbackMode?: 'video' | 'frames'
    framesDirName: string | null
    rawDataFileName: string
    fieldMapDataFileName: string | null
    airProfileDataFileName?: string | null
    stats: OverlayStats
  } | null
  media: {
    videoUrl: string | null
    overlayUrl: string | null
    overlayTransparentUrl: string | null
    overlayVideoUrl: string | null
    overlayFrameUrlTemplate: string | null
    fieldMapDataUrl: string | null
    airProfileDataUrl?: string | null
  }
  lastError: string | null
  processingProgress: ProcessingProgress | null
}

export type DisplayMode = 'match' | 'field'

export type StreamClipperOcrRegion = {
  x: number
  y: number
  width: number
  height: number
}

export type StreamClipperConfig = {
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

export type StreamClipperDetection = {
  seconds: number
  text: string
  confidence: number
  detectedAt: string
  region: string
}

export type StreamClipperActiveMatch = {
  startedAt: string | null
  secondsLeft: number | null
  statusText: string
}

export type StreamClipperStatus = {
  running: boolean
  phase: string
  matchState: string
  streamUrl: string
  resolvedStreamUrl: string | null
  startedAt: string | null
  updatedAt: string
  lastError: string | null
  workDir: string
  videosDir: string
  config: StreamClipperConfig
  bufferedSeconds: number
  bufferedSegments: number
  latestDetection: StreamClipperDetection | null
  recentClips: string[]
  activeMatch: StreamClipperActiveMatch
}
