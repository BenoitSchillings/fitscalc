#!/usr/bin/env python3
"""Interactive FITS image calculator.

Bare names refer to <name>.fits in the current directory.

    fits> a = b + c              writes a.fits = b.fits + c.fits
    fits> d = (b - c) * 0.5
    fits> e = sqrt(b)
    fits> a = mean(images*)      per-pixel mean over images*.fits
    fits> a = median(img1, img2, img3)
    fits> a = sum("dark/*.fits")
    fits> a = mean(planet.ser)   per-pixel mean over all frames in a SER file
    fits> count(planet.ser)      number of frames in a SER file
    fits> a = mean_sigma(lights*, k=3)   sigma-clipped stack
    fits> count(lights*)         number of files matching lights*.fits
    fits> percentile(b, 99.5)    99.5th percentile of b
    fits> sub = crop(b, 100, 200, 600, 900)   x1,y1,x2,y2 (end-exclusive)
    fits> bg(b)                  sigma-clipped sky-background value
    fits> bgnoise(b)             sigma-clipped sky-noise (stddev)
    fits> flat = b - bg(b)       subtract scalar background
    fits> back = bg2d(b, box=64) 2D background (gradient/vignetting) map
    fits> flat = b - bg2d(b, box=64)
    fits> debanded = clean(b)    remove column-fixed vertical banding
    fits> debanded = clean(b, kernel=128)
    fits> view(b)                open b.fits in the image viewer
    fits> view(lights*)          browse a sequence (Prev/Next, A=auto-scale)
    fits> b                      print stats for b.fits
    fits> ls                     list *.fits and *.ser in cwd
    fits> ls ..                  list FITS/SER in another directory
    fits> ls *.fits              glob for matching files (any extension)
    fits> info b                 show FITS HDU info
    fits> stats b
    fits> quit
"""
import ast
import contextlib
import glob as globmod
import io
import json
import os
import re
import socket
import subprocess
import sys
import time
import readline  # noqa: F401  enables line editing / history

import numpy as np
from astropy.io import fits as afits
from astropy.stats import sigma_clip as _sigma_clip, sigma_clipped_stats


EXT = ".fits"

# A glob token: contains *, ?, or [, sits inside a (...) or after a comma.
# Lookbehind/ahead make sure we don't grab a stray "*" from b * c.
GLOB_CHARS = r"\w./\-*?\[\]"
GLOB_RE = re.compile(
    rf"(?<=[(,=])\s*([{GLOB_CHARS}]*[*?\[][{GLOB_CHARS}]*)\s*(?=[,)]|$)"
)
# Bare .ser paths: filename ending in .ser (no glob meta) in argument-list position.
SER_RE = re.compile(r"(?<=[(,=])\s*([A-Za-z0-9_./\-]+\.ser)\s*(?=[,)]|$)")


def preprocess(line: str) -> str:
    """Rewrite bare glob and .ser tokens into string literals so ast.parse accepts them."""
    line = SER_RE.sub(lambda m: repr(m.group(1)), line)
    line = GLOB_RE.sub(lambda m: repr(m.group(1)), line)
    return line


def path_of(name: str) -> str:
    return name + EXT


def load_path(path: str) -> np.ndarray:
    if not os.path.exists(path) and not path.endswith(EXT) and os.path.exists(path + EXT):
        path = path + EXT
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with afits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=np.float64)
    raise ValueError(f"{path} has no image data")


def load(name: str) -> np.ndarray:
    return load_path(path_of(name))


def _glob_paths(pattern: str) -> list[str]:
    full = pattern if pattern.endswith(EXT) or "." in os.path.basename(pattern) else pattern + EXT
    return sorted(globmod.glob(full))


def _resolve_paths(node) -> list[str]:
    """Resolve a view() argument node to a list of filesystem paths (no loading)."""
    if isinstance(node, ast.Name):
        return [path_of(node.id)]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        s = node.value
        if any(c in s for c in "*?["):
            paths = _glob_paths(s)
            if not paths:
                raise FileNotFoundError(f"no files match {s}")
            return paths
        if os.path.exists(s):
            return [s]
        if not s.endswith(EXT) and os.path.exists(s + EXT):
            return [s + EXT]
        raise FileNotFoundError(f"{s} not found")
    raise ValueError(f"view() arg must be a name or quoted path/glob, got {ast.unparse(node)}")


VIEWER_SOCK_PATH = f"/tmp/fitscalc-viewer-{os.getuid()}.sock"


def _try_send_to_viewer(files: list[str]) -> bool:
    """Send file list to a running viewer over its IPC socket. Returns True on success."""
    if not os.path.exists(VIEWER_SOCK_PATH):
        return False
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.5)
        s.connect(VIEWER_SOCK_PATH)
        s.sendall(json.dumps({"files": files}).encode())
        s.shutdown(socket.SHUT_WR)
        return True
    except (OSError, socket.timeout):
        # Stale socket file (viewer crashed): remove it so the spawn path runs cleanly.
        try:
            os.unlink(VIEWER_SOCK_PATH)
        except OSError:
            pass
        return False
    finally:
        s.close()


def _launch_viewer(files: list[str]) -> None:
    abs_files = [os.path.abspath(p) for p in files]
    if _try_send_to_viewer(abs_files):
        plural = "s" if len(abs_files) != 1 else ""
        print(f"sent {len(abs_files)} file{plural} to running viewer")
        return
    here = os.path.dirname(os.path.abspath(__file__))
    viewer = os.environ.get("FITSCALC_VIEWER", os.path.join(here, "view.py"))
    if not os.path.exists(viewer):
        raise FileNotFoundError(f"viewer not found: {viewer}")
    subprocess.Popen(
        [sys.executable, viewer, *abs_files],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Wait briefly for the viewer's IPC socket to appear so a quick follow-up
    # view() call reuses this instance instead of spawning a second one.
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline and not os.path.exists(VIEWER_SOCK_PATH):
        time.sleep(0.05)
    plural = "s" if len(abs_files) != 1 else ""
    print(f"opened viewer ({len(abs_files)} file{plural})")


def _open_ser(path: str):
    """Open a SER file, suppressing the library's debug print on construction."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    from ser import Ser
    with contextlib.redirect_stdout(io.StringIO()):
        return Ser(path)


def _load_ser_frames(path: str) -> list[np.ndarray]:
    s = _open_ser(path)
    try:
        frames = []
        for i in range(int(s.count)):
            img = s.load_img(i)
            if img is None:
                continue
            frames.append(np.asarray(img, dtype=np.float64))
        if not frames:
            raise ValueError(f"{path}: no frames could be loaded")
        return frames
    finally:
        s.close()


def _ser_frame_count(path: str) -> int:
    s = _open_ser(path)
    try:
        return int(s.count)
    finally:
        s.close()


def _load_one(path: str):
    """Load a single path - .ser expands to list of frames; otherwise FITS."""
    if path.endswith(".ser"):
        return _load_ser_frames(path)
    return load_path(path)


def expand_string(s: str):
    """Resolve a string operand: glob -> list of arrays, plain path -> array (or list, for .ser)."""
    if any(c in s for c in "*?["):
        paths = _glob_paths(s)
        if not paths:
            raise FileNotFoundError(f"no files match {s}")
        out = []
        for p in paths:
            loaded = _load_one(p)
            if isinstance(loaded, list):
                out.extend(loaded)
            else:
                out.append(loaded)
        return out
    return _load_one(s)


def save(name: str, data: np.ndarray) -> None:
    path = path_of(name)
    afits.writeto(path, data.astype(np.float32), overwrite=True)


def _collect_stack(args):
    items = []
    for a in args:
        if isinstance(a, (list, tuple)):
            items.extend(a)
        else:
            items.append(a)
    if len(items) < 2:
        raise ValueError("stack reduction needs at least 2 frames")
    return np.stack([np.asarray(x) for x in items], axis=0)


def _stack_reduce(reduce_fn):
    """One ndarray arg -> scalar reduce; list arg or multiple args -> per-pixel reduce."""
    def fn(*args):
        if len(args) == 1 and not isinstance(args[0], (list, tuple)):
            return reduce_fn(np.asarray(args[0]))
        return reduce_fn(_collect_stack(args), axis=0)
    return fn


def _sigma_reduce(masked_reduce):
    def fn(*args, k=3.0, maxiters=5):
        stack = _collect_stack(args)
        clipped = _sigma_clip(
            stack, sigma=float(k), maxiters=int(maxiters), axis=0, masked=True,
            cenfunc="median", stdfunc="mad_std",
        )
        result = masked_reduce(clipped, axis=0)
        if hasattr(result, "filled"):
            result = result.filled(np.nan)
        return np.asarray(result)
    return fn


def _percentile(img, p):
    return float(np.percentile(np.asarray(img), float(p)))


def _crop(img, x1, y1, x2, y2):
    a = np.asarray(img)
    return a[int(y1):int(y2), int(x1):int(x2)]


def _bg(img, k=3.0, maxiters=5):
    a = np.asarray(img)
    if a.ndim < 2:
        return float(a)
    _, median, _ = sigma_clipped_stats(a, sigma=float(k), maxiters=int(maxiters))
    return float(median)


def _bgnoise(img, k=3.0, maxiters=5):
    a = np.asarray(img)
    if a.ndim < 2:
        return float(np.std(a)) if a.size > 1 else 0.0
    _, _, std = sigma_clipped_stats(a, sigma=float(k), maxiters=int(maxiters))
    return float(std)


def _clean(img, kernel=64):
    """Remove vertical column banding by dividing out the median per-column gain.

    For each row, divide by a horizontal Gaussian smooth (sigma = kernel/4)
    to get the high-frequency residual. Real structure varies row-to-row and
    cancels in the median across rows; column-fixed banding survives.
    """
    from scipy.ndimage import gaussian_filter1d
    a = np.asarray(img, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("clean() expects a 2D image")
    ksize = int(kernel)
    if ksize % 2 == 0:
        ksize += 1
    sigma = ksize / 4.0
    smoothed = gaussian_filter1d(a, sigma=sigma, axis=1, mode="reflect")
    smoothed = np.maximum(smoothed, 1e-10)
    ratio = a / smoothed
    banding = np.median(ratio, axis=0)
    banding = np.clip(banding, 0.5, 2.0)
    correction = 1.0 / banding
    return a * correction[np.newaxis, :]


def _bg2d(img, box=64, filter_size=3, mask=None):
    from photutils.background import Background2D, MedianBackground
    a = np.asarray(img, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) if mask is not None else None
    bkg = Background2D(
        a,
        box_size=int(box),
        filter_size=int(filter_size),
        bkg_estimator=MedianBackground(),
        mask=m,
    )
    return np.asarray(bkg.background)


FUNCS = {
    "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "log2": np.log2,
    "exp": np.exp, "abs": np.abs, "clip": np.clip, "where": np.where,
    "mean": _stack_reduce(np.mean),
    "median": _stack_reduce(np.median),
    "std": _stack_reduce(np.std),
    "min": _stack_reduce(np.min),
    "max": _stack_reduce(np.max),
    "sum": _stack_reduce(np.sum),
    "mean_sigma": _sigma_reduce(np.ma.mean),
    "median_sigma": _sigma_reduce(np.ma.median),
    "percentile": _percentile,
    "crop": _crop,
    "bg": _bg,
    "bg2d": _bg2d,
    "bgnoise": _bgnoise,
    "clean": _clean,
}

BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}

UNARYOPS = {ast.UAdd: lambda a: +a, ast.USub: lambda a: -a}

CMPOPS = {
    ast.Lt: np.less, ast.LtE: np.less_equal,
    ast.Gt: np.greater, ast.GtE: np.greater_equal,
    ast.Eq: np.equal, ast.NotEq: np.not_equal,
}


def evaluate(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return expand_string(node.value)
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        return load(node.id)
    if isinstance(node, ast.BinOp):
        op = BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported operator {type(node.op).__name__}")
        return op(evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp):
        op = UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary op {type(node.op).__name__}")
        return op(evaluate(node.operand))
    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError("only single comparisons are supported")
        op = CMPOPS.get(type(node.ops[0]))
        if op is None:
            raise ValueError(f"unsupported comparison {type(node.ops[0]).__name__}")
        return op(evaluate(node.left), evaluate(node.comparators[0]))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError(f"unknown function: {ast.unparse(node.func)}")
        fname = node.func.id
        if fname == "count":
            if (len(node.args) != 1 or node.keywords
                    or not isinstance(node.args[0], ast.Constant)
                    or not isinstance(node.args[0].value, str)):
                raise ValueError("count() takes a glob or .ser path: count(images*) or count(planet.ser)")
            s = node.args[0].value
            if s.endswith(".ser") and not any(c in s for c in "*?["):
                return _ser_frame_count(s)
            return len(_glob_paths(s))
        if fname == "view":
            if not node.args or node.keywords:
                raise ValueError("view() takes one or more file references: view(b), view(lights*)")
            files = []
            for a in node.args:
                files.extend(_resolve_paths(a))
            _launch_viewer(files)
            return None
        if fname not in FUNCS:
            raise ValueError(f"unknown function: {fname}")
        args = [evaluate(a) for a in node.args]
        kwargs = {kw.arg: evaluate(kw.value) for kw in node.keywords}
        return FUNCS[fname](*args, **kwargs)
    raise ValueError(f"unsupported expression: {ast.unparse(node)}")


def stats_str(arr) -> str:
    a = np.asarray(arr)
    if a.ndim == 0:
        return f"scalar={a.item():.6g}"
    return (
        f"shape={a.shape} dtype={a.dtype} "
        f"min={np.nanmin(a):.6g} max={np.nanmax(a):.6g} "
        f"mean={np.nanmean(a):.6g} std={np.nanstd(a):.6g}"
    )


def cmd_ls(args):
    """List FITS/SER files. With no args: cwd. With args: each may be a
    directory (lists FITS/SER inside), a glob (any extension), or a file."""
    suffixes = (EXT, ".ser")

    if not args:
        for p in sorted(p for p in os.listdir(".") if p.endswith(suffixes)):
            print(p)
        return

    multi = len(args) > 1
    for i, raw in enumerate(args):
        t = os.path.expanduser(raw)
        if multi and i > 0:
            print()
        if os.path.isdir(t):
            if multi:
                print(f"{raw}:")
            try:
                entries = sorted(p for p in os.listdir(t) if p.endswith(suffixes))
            except OSError as e:
                print(f"ls: {raw}: {e}", file=sys.stderr)
                continue
            for e in entries:
                print(e)
            continue
        matches = sorted(globmod.glob(t))
        if not matches:
            if any(c in t for c in "*?["):
                print(f"ls: no match for '{raw}'", file=sys.stderr)
            else:
                print(f"ls: cannot access '{raw}': No such file or directory",
                      file=sys.stderr)
            continue
        if multi:
            print(f"{raw}:")
        for m in matches:
            print(m)


def cmd_info(name):
    with afits.open(path_of(name)) as hdul:
        hdul.info()


def run_line(line: str) -> None:
    line = line.strip()
    if not line or line.startswith("#"):
        return
    parts = line.split()
    head = parts[0]
    if head in ("quit", "exit", "q"):
        raise SystemExit(0)
    if head == "help":
        print(__doc__)
        return
    if head == "ls":
        cmd_ls(parts[1:])
        return
    if head == "info" and len(parts) == 2:
        cmd_info(parts[1])
        return
    if head == "stats" and len(parts) == 2:
        print(stats_str(load(parts[1])))
        return

    tree = ast.parse(preprocess(line), mode="exec")
    if len(tree.body) != 1:
        raise ValueError("one statement per line")
    stmt = tree.body[0]
    if isinstance(stmt, ast.Assign):
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            raise ValueError("assignment target must be a bare name")
        name = stmt.targets[0].id
        result = evaluate(stmt.value)
        if result is None:
            raise ValueError("view() returns nothing - cannot assign")
        if isinstance(result, (list, tuple)):
            raise ValueError(
                f"got {len(result)} images; wrap in a reduction "
                "(mean/median/sum/min/max/std) to write a single FITS"
            )
        arr = np.asarray(result)
        if arr.ndim == 0:
            print(f"{name} = {arr.item():.6g}")
            return
        save(name, arr)
        print(f"wrote {path_of(name)}  {stats_str(arr)}")
    elif isinstance(stmt, ast.Expr):
        result = evaluate(stmt.value)
        if result is None:
            return
        print(stats_str(result))
    else:
        raise ValueError("expected an expression or assignment")


def repl():
    print("fits calculator -- try:  a = b + c   |  ls  |  info b  |  help  |  quit")
    while True:
        try:
            line = input("fits> ")
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            continue
        try:
            run_line(line)
        except SystemExit:
            return
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_line(" ".join(sys.argv[1:]))
    else:
        repl()
