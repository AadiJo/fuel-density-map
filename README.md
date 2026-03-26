# fuel-density-map

Local SPA for importing a YouTube match clip, drawing the field region to analyze, generating a fuel-density overlay, and reviewing saved sessions on your machine.

## Requirements

- [Bun](https://bun.sh/)
- Python 3.14+
- Python packages from `requirements.txt`

## Install

```powershell
python -m pip install -r requirements.txt
bun install
cd webui
bun install
cd ..
```

## Run in Development

```powershell
bun run dev
```

That starts:

- the Bun API on `http://localhost:3001`
- the Vite SPA on `http://localhost:5173`

## Build for Local Use

```powershell
bun run build
bun run start
```

`bun run start` serves the built SPA and the API from one local server.

## Storage

- Sessions are stored on disk under `sessions/`
- Each session keeps its downloaded video, generated overlay PNGs, raw data, and session metadata locally

## Notes

- The YouTube import flow uses `yt-dlp` through Python, installed from `requirements.txt`
- The processing pipeline still uses OpenCV, NumPy, and Pillow
