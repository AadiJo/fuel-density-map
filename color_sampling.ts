export type RgbColor = {
  r: number
  g: number
  b: number
}

export function clamp01(value: number) {
  if (!Number.isFinite(value)) {
    return 0
  }
  return Math.min(Math.max(value, 0), 1)
}

export function normalizedPointToPixel(x: number, y: number, width: number, height: number) {
  const safeWidth = Math.max(1, Math.floor(width))
  const safeHeight = Math.max(1, Math.floor(height))
  const px = Math.max(0, Math.min(safeWidth - 1, Math.round(clamp01(x) * (safeWidth - 1))))
  const py = Math.max(0, Math.min(safeHeight - 1, Math.round(clamp01(y) * (safeHeight - 1))))
  return { px, py }
}

export function rgbFromRawRgb24Frame(frame: Uint8Array, width: number, height: number, px: number, py: number): RgbColor {
  const safeWidth = Math.max(1, Math.floor(width))
  const safeHeight = Math.max(1, Math.floor(height))
  const frameBytes = safeWidth * safeHeight * 3
  if (frame.length < frameBytes) {
    throw new Error('Decoded frame is incomplete.')
  }
  const safeX = Math.max(0, Math.min(safeWidth - 1, Math.floor(px)))
  const safeY = Math.max(0, Math.min(safeHeight - 1, Math.floor(py)))
  const offset = (safeY * safeWidth + safeX) * 3
  return {
    r: frame[offset],
    g: frame[offset + 1],
    b: frame[offset + 2],
  }
}

export function sampleRgbFromRawRgb24Frame(
  frame: Uint8Array,
  width: number,
  height: number,
  px: number,
  py: number,
  radius = 2,
): RgbColor {
  const safeWidth = Math.max(1, Math.floor(width))
  const safeHeight = Math.max(1, Math.floor(height))
  const frameBytes = safeWidth * safeHeight * 3
  if (frame.length < frameBytes) {
    throw new Error('Decoded frame is incomplete.')
  }

  const safeX = Math.max(0, Math.min(safeWidth - 1, Math.floor(px)))
  const safeY = Math.max(0, Math.min(safeHeight - 1, Math.floor(py)))
  const sampleRadius = Math.max(0, Math.floor(radius))
  const startX = Math.max(0, safeX - sampleRadius)
  const endX = Math.min(safeWidth - 1, safeX + sampleRadius)
  const startY = Math.max(0, safeY - sampleRadius)
  const endY = Math.min(safeHeight - 1, safeY + sampleRadius)

  const rs: number[] = []
  const gs: number[] = []
  const bs: number[] = []
  for (let y = startY; y <= endY; y += 1) {
    for (let x = startX; x <= endX; x += 1) {
      const offset = (y * safeWidth + x) * 3
      rs.push(frame[offset])
      gs.push(frame[offset + 1])
      bs.push(frame[offset + 2])
    }
  }

  const midpoint = Math.floor(rs.length / 2)
  rs.sort((a, b) => a - b)
  gs.sort((a, b) => a - b)
  bs.sort((a, b) => a - b)
  return {
    r: rs[midpoint],
    g: gs[midpoint],
    b: bs[midpoint],
  }
}
