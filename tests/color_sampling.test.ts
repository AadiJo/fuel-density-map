import { describe, expect, test } from 'bun:test'

import { normalizedPointToPixel, rgbFromRawRgb24Frame, sampleRgbFromRawRgb24Frame } from '../color_sampling'

describe('normalizedPointToPixel', () => {
  test('clamps normalized coordinates and maps to edge pixels', () => {
    expect(normalizedPointToPixel(-2, -1, 10, 8)).toEqual({ px: 0, py: 0 })
    expect(normalizedPointToPixel(2, 4, 10, 8)).toEqual({ px: 9, py: 7 })
  })

  test('maps center-ish coordinates consistently', () => {
    expect(normalizedPointToPixel(0.5, 0.5, 5, 5)).toEqual({ px: 2, py: 2 })
  })
})

describe('rgbFromRawRgb24Frame', () => {
  test('extracts expected RGB values from a raw frame', () => {
    // 2x2 frame laid out row-major in RGB24
    // (0,0)=10,20,30 (1,0)=40,50,60
    // (0,1)=70,80,90 (1,1)=100,110,120
    const frame = new Uint8Array([
      10, 20, 30,
      40, 50, 60,
      70, 80, 90,
      100, 110, 120,
    ])

    expect(rgbFromRawRgb24Frame(frame, 2, 2, 0, 0)).toEqual({ r: 10, g: 20, b: 30 })
    expect(rgbFromRawRgb24Frame(frame, 2, 2, 1, 0)).toEqual({ r: 40, g: 50, b: 60 })
    expect(rgbFromRawRgb24Frame(frame, 2, 2, 0, 1)).toEqual({ r: 70, g: 80, b: 90 })
    expect(rgbFromRawRgb24Frame(frame, 2, 2, 1, 1)).toEqual({ r: 100, g: 110, b: 120 })
  })

  test('throws on incomplete frame buffers', () => {
    const frame = new Uint8Array([1, 2, 3])
    expect(() => rgbFromRawRgb24Frame(frame, 2, 2, 0, 0)).toThrow('Decoded frame is incomplete.')
  })
})

describe('sampleRgbFromRawRgb24Frame', () => {
  test('returns the median color from a small neighborhood', () => {
    const frame = new Uint8Array([
      10, 20, 30,
      12, 22, 32,
      240, 240, 10,
      14, 24, 34,
      16, 26, 36,
      18, 28, 38,
      20, 30, 40,
      22, 32, 42,
      24, 34, 44,
    ])

    expect(sampleRgbFromRawRgb24Frame(frame, 3, 3, 1, 1, 1)).toEqual({ r: 18, g: 28, b: 38 })
  })

  test('clamps the sample window at the frame edge', () => {
    const frame = new Uint8Array([
      5, 10, 15,
      200, 210, 220,
      7, 12, 17,
      9, 14, 19,
    ])

    expect(sampleRgbFromRawRgb24Frame(frame, 2, 2, 0, 0, 1)).toEqual({ r: 9, g: 14, b: 19 })
  })
})
