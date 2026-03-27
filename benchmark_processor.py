import argparse
import json
import os
import shutil
import subprocess
import tempfile
import time


def run_once(args):
    work_dir = tempfile.mkdtemp(prefix="fdm-bench-")
    try:
        cmd = [
            "python",
            "processor_cli.py",
            "--video",
            args.video,
            "--session-dir",
            work_dir,
            "--bbox",
            args.bbox,
            "--backend",
            args.backend,
            "--overlay-output",
            args.overlay_output,
            "--target-process-fps",
            str(args.target_process_fps),
            "--working-scale",
            str(args.working_scale),
            "--max-active-tracks",
            str(args.max_active_tracks),
        ]
        if args.detector_budget:
            cmd.extend(["--detector-budget", str(args.detector_budget)])
        if args.quad:
            cmd.extend(["--quad", args.quad])

        started = time.perf_counter()
        subprocess.run(cmd, check=True)
        elapsed = time.perf_counter() - started

        with open(os.path.join(work_dir, "stats.json"), "r", encoding="utf-8") as handle:
            stats = json.load(handle)
        return elapsed, stats
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the fuel-density processor.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--bbox", required=True, help="x,y,width,height")
    parser.add_argument("--quad", help="x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--overlay-output", choices=("video", "frames"), default="video")
    parser.add_argument("--target-process-fps", type=float, default=15.0)
    parser.add_argument("--working-scale", type=float, default=0.75)
    parser.add_argument("--detector-budget", type=int, default=0)
    parser.add_argument("--max-active-tracks", type=int, default=900)
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    results = []
    for index in range(max(1, args.runs)):
        elapsed, stats = run_once(args)
        results.append((elapsed, stats))
        timings = stats.get("timings", {})
        print(
            f"run={index + 1} elapsed={elapsed:.2f}s "
            f"backend={stats.get('backend', 'unknown')} "
            f"decode={timings.get('decode', 0):.2f}s "
            f"mask={timings.get('mask', 0):.2f}s "
            f"detect={timings.get('detect', 0):.2f}s "
            f"stabilize={timings.get('stabilize', 0):.2f}s "
            f"project={timings.get('project', 0):.2f}s "
            f"encode={timings.get('encode', 0):.2f}s"
        )

    elapsed_values = [elapsed for elapsed, _ in results]
    best = min(elapsed_values)
    mean = sum(elapsed_values) / len(elapsed_values)
    print(f"best={best:.2f}s mean={mean:.2f}s runs={len(elapsed_values)}")


if __name__ == "__main__":
    main()
