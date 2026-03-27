#!/usr/bin/env python3
import argparse
import os
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path


DEFAULT_OPENCV_VERSION = "4.12.0"
DEFAULT_CUDA_ARCH_BIN = "8.9"


def run(cmd, *, cwd=None, env=None):
    print("+", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def python_site_packages(python_bin: Path) -> str:
    output = subprocess.check_output(
        [str(python_bin), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    )
    return output.strip()


def python_numpy_include(python_bin: Path) -> str:
    output = subprocess.check_output(
        [str(python_bin), "-c", "import numpy; print(numpy.get_include())"],
        text=True,
    )
    return output.strip()


def ensure_venv(venv_dir: Path):
    python_bin = venv_dir / "bin" / "python"
    if not python_bin.exists():
        run([sys.executable, "-m", "venv", str(venv_dir)])
    return python_bin


def ensure_sources(repo_root: Path, version: str, bootstrap: bool):
    third_party = repo_root / "third_party"
    third_party.mkdir(exist_ok=True)
    sources = {
        "opencv": f"https://github.com/opencv/opencv.git",
        "opencv_contrib": f"https://github.com/opencv/opencv_contrib.git",
    }
    for name, url in sources.items():
        target = third_party / name
        if (target / ".git").exists():
            continue
        if not bootstrap:
            raise SystemExit(
                f"Missing {target}. Re-run with --bootstrap once network access is available."
            )
        run(["git", "clone", "--branch", version, "--depth", "1", url, str(target)])


def bootstrap_build_env(repo_root: Path, build_python: Path):
    build_env = os.environ.copy()
    run(
        [
            str(build_python),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "setuptools",
            "wheel",
            "cmake",
            "ninja",
        ],
        cwd=repo_root,
        env=build_env,
    )


def ensure_build_tools(build_venv: Path, repo_root: Path, bootstrap: bool):
    build_python = build_venv / "bin" / "python"
    cmake_bin = build_venv / "bin" / "cmake"
    ninja_bin = build_venv / "bin" / "ninja"
    if cmake_bin.exists() and ninja_bin.exists() and build_python.exists():
        return build_python
    if not bootstrap:
        raise SystemExit(
            "Missing .venv-opencv-cuda-build with cmake/ninja. Re-run with --bootstrap once network access is available."
        )
    build_python = ensure_venv(build_venv)
    bootstrap_build_env(repo_root, build_python)
    return build_python


def symlink_package_tree(src_root: Path, dst_root: Path, package_name: str):
    patterns = [package_name, f"{package_name}*.dist-info", f"{package_name}*.data", f"{package_name}.libs"]
    matched = False
    for pattern in patterns:
        for source in sorted(src_root.glob(pattern)):
            target = dst_root / source.name
            if target.exists() or target.is_symlink():
                matched = True
                continue
            target.symlink_to(source)
            matched = True
    return matched


def sync_runtime_python_packages(runtime_python: Path):
    runtime_site = Path(python_site_packages(runtime_python))
    runtime_site.mkdir(parents=True, exist_ok=True)

    user_site = Path(site.getusersitepackages())
    if not user_site.exists():
        raise SystemExit(f"User site-packages not found at {user_site}")

    required = {
        "numpy": "NumPy",
        "PIL": "Pillow",
    }
    for package_name, label in required.items():
        if symlink_package_tree(user_site, runtime_site, package_name):
            continue
        raise SystemExit(
            f"{label} is not available in {user_site}. Install it in your normal Python first, then rerun this script."
        )
    return str(runtime_site)


def main():
    parser = argparse.ArgumentParser(description="Build a repo-local CUDA-enabled OpenCV Python runtime.")
    parser.add_argument("--opencv-version", default=DEFAULT_OPENCV_VERSION)
    parser.add_argument("--cuda-root", default="/usr/local/cuda")
    parser.add_argument("--cuda-arch-bin", default=DEFAULT_CUDA_ARCH_BIN)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 8)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    build_venv = repo_root / ".venv-opencv-cuda-build"
    runtime_venv = repo_root / ".venv-opencv-cuda"
    build_dir = repo_root / ".build" / "opencv-cuda"
    install_dir = repo_root / ".opencv-cuda-install"

    cuda_root = Path(args.cuda_root)
    nvcc = cuda_root / "bin" / "nvcc"
    if not nvcc.exists():
        raise SystemExit(f"CUDA compiler not found at {nvcc}")

    build_python = ensure_build_tools(build_venv, repo_root, bootstrap=args.bootstrap)
    runtime_python = ensure_venv(runtime_venv)

    ensure_sources(repo_root, args.opencv_version, bootstrap=args.bootstrap)

    if args.clean:
        shutil.rmtree(build_dir, ignore_errors=True)
        shutil.rmtree(install_dir, ignore_errors=True)

    build_dir.mkdir(parents=True, exist_ok=True)
    install_dir.mkdir(parents=True, exist_ok=True)

    runtime_site = sync_runtime_python_packages(runtime_python)
    runtime_numpy_include = python_numpy_include(runtime_python)
    python_include = sysconfig.get_paths()["include"]
    python_library = sysconfig.get_config_var("LIBDIR")
    python_ldlibrary = sysconfig.get_config_var("LDLIBRARY")
    if not python_library or not python_ldlibrary:
        raise SystemExit("Unable to determine Python library path for the OpenCV build.")

    build_env = os.environ.copy()
    build_env["PATH"] = f"{build_venv / 'bin'}:{cuda_root / 'bin'}:{build_env.get('PATH', '')}"
    build_env["CUDA_HOME"] = str(cuda_root)
    build_env["CUDA_PATH"] = str(cuda_root)
    build_env["PYTHONNOUSERSITE"] = "1"

    cmake_args = [
        str(build_venv / "bin" / "cmake"),
        "-S",
        str(repo_root / "third_party" / "opencv"),
        "-B",
        str(build_dir),
        "-G",
        "Ninja",
        "-D",
        "CMAKE_BUILD_TYPE=Release",
        "-D",
        f"CMAKE_INSTALL_PREFIX={install_dir}",
        "-D",
        f"OPENCV_EXTRA_MODULES_PATH={repo_root / 'third_party' / 'opencv_contrib' / 'modules'}",
        "-D",
        "BUILD_LIST=core,imgproc,imgcodecs,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping",
        "-D",
        "BUILD_SHARED_LIBS=ON",
        "-D",
        "BUILD_TESTS=OFF",
        "-D",
        "BUILD_PERF_TESTS=OFF",
        "-D",
        "BUILD_EXAMPLES=OFF",
        "-D",
        "BUILD_JAVA=OFF",
        "-D",
        "BUILD_opencv_apps=OFF",
        "-D",
        "BUILD_opencv_gapi=OFF",
        "-D",
        "BUILD_opencv_world=OFF",
        "-D",
        "BUILD_opencv_highgui=OFF",
        "-D",
        "BUILD_opencv_videoio=OFF",
        "-D",
        "WITH_CUDA=ON",
        "-D",
        "WITH_IPP=OFF",
        "-D",
        "WITH_CUDNN=OFF",
        "-D",
        "OPENCV_DNN_CUDA=OFF",
        "-D",
        "WITH_CUBLAS=ON",
        "-D",
        "ENABLE_FAST_MATH=ON",
        "-D",
        "CUDA_FAST_MATH=ON",
        "-D",
        "WITH_FFMPEG=OFF",
        "-D",
        "WITH_GSTREAMER=OFF",
        "-D",
        "WITH_QT=OFF",
        "-D",
        "WITH_GTK=OFF",
        "-D",
        f"CUDA_ARCH_BIN={args.cuda_arch_bin}",
        "-D",
        f"CUDA_TOOLKIT_ROOT_DIR={cuda_root}",
        "-D",
        f"CMAKE_CUDA_COMPILER={nvcc}",
        "-D",
        f"Python3_EXECUTABLE={runtime_python}",
        "-D",
        f"PYTHON3_EXECUTABLE={runtime_python}",
        "-D",
        f"PYTHON3_INCLUDE_DIR={python_include}",
        "-D",
        f"PYTHON3_LIBRARY={Path(python_library) / python_ldlibrary}",
        "-D",
        f"PYTHON3_NUMPY_INCLUDE_DIRS={runtime_numpy_include}",
        "-D",
        f"OPENCV_PYTHON3_INSTALL_PATH={runtime_site}",
    ]

    run(cmake_args, cwd=repo_root, env=build_env)
    run(
        [str(build_venv / "bin" / "cmake"), "--build", str(build_dir), "--parallel", str(args.jobs)],
        cwd=repo_root,
        env=build_env,
    )
    run(
        [str(build_venv / "bin" / "cmake"), "--install", str(build_dir)],
        cwd=repo_root,
        env=build_env,
    )

    cv2_info = subprocess.check_output(
        [
            str(runtime_python),
            "-c",
            "import cv2; print(cv2.__file__); print(cv2.getBuildInformation())",
        ],
        text=True,
        env=build_env,
    )
    print(cv2_info)
    print(f"Use this interpreter for the processor: {runtime_python}")


if __name__ == "__main__":
    main()
