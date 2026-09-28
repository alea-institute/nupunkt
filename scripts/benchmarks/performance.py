#!/usr/bin/env python
"""Reproducible speed / footprint benchmark for nupunkt (and, optionally, comparators).

Stdlib only.  Default mode benchmarks the nupunkt importable by the running interpreter
(e.g. the project checkout) plus a regex floor:

    uv run python scripts/benchmarks/performance.py

``--with-comparators`` additionally builds throwaway ``uv`` venvs (one per library) for
nupunkt 0.7.0 (local wheel if present, else PyPI), 0.6.0, 0.5.1, nltk (punkt_tab), pysbd,
sentencex, blingfire, spaCy (blank ``en`` + sentencizer) and syntok, and runs every
measurement in fresh subprocesses of those venvs:

    python scripts/benchmarks/performance.py --with-comparators --json results.json

Every timing is the median of ``--runs`` (default 5) repetitions.  Each (library, task)
runs in its own fresh process, pinned to one core with ``taskset -c --cpu`` when
available.  Corpora are built deterministically into ``--work-dir``:
  * Project Gutenberg #11 (Alice) and #1661 (Sherlock Holmes), headers/footers stripped;
  * from ``data/test.jsonl.gz`` (seeded shuffle, docs joined by blank lines): a ~4 MB
    legal corpus, all 38,527 docs (14.5M chars, no repeats), and ~50 MB made by
    repeating the latter 3.4 times;
  * the 38,527 individual documents of ``data/test.jsonl.gz`` for per-document calls.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GUTENBERG = {"alice": 11, "sherlock": 1661}

# Source of the per-library splitter factory.  It is exec'd inside fresh interpreters so
# that "import + first call" is measured without this script's own import overhead.
FACTORY = r"""
def make(lib):
    if lib == "regex":
        import re
        return re.compile(r"(?<=[.!?])\s+").split
    if lib == "nupunkt":
        import nupunkt
        return nupunkt.sent_tokenize
    if lib == "nupunkt-adaptive":
        import nupunkt
        return lambda t: nupunkt.sent_tokenize(t, adaptive=True)
    if lib == "nltk":
        from nltk.tokenize import sent_tokenize
        return sent_tokenize
    if lib == "pysbd":
        import pysbd
        return pysbd.Segmenter(language="en", clean=False).segment
    if lib == "sentencex":
        import sentencex
        return lambda t: list(sentencex.segment("en", t))
    if lib == "blingfire":
        import blingfire
        return lambda t: blingfire.text_to_sentences(t).split("\n")
    if lib == "spacy":
        import spacy
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        nlp.max_length = 10**9
        return lambda t: [s.text for s in nlp(t).sents]
    if lib == "syntok":
        from syntok import segmenter
        return lambda t: ["".join(k.spacing + k.value for k in s)
                          for p in segmenter.process(t) for s in p]
    raise ValueError(lib)


def reset(lib):
    # nupunkt >= 0.7.0 memoizes boundary decisions per context string; clearing the memo
    # gives "unseen text" timings when the same input is tokenized repeatedly.
    if lib == "nupunkt":
        import nupunkt
        tok = nupunkt.load("default") if hasattr(nupunkt, "load") else None
        if hasattr(tok, "clear_decision_cache"):
            tok.clear_decision_cache()
            return True
    return False
"""

COLD = (
    FACTORY
    + r"""
import json, sys, time
lib, path = sys.argv[1], sys.argv[2]
text = open(path, encoding="utf-8").read()
t0 = time.perf_counter()
imp = None
if lib.startswith("nupunkt"):
    import nupunkt
    imp = time.perf_counter() - t0
fn = make(lib)
fn(text)
total = time.perf_counter() - t0
# VmHWM, not ru_maxrss: the latter survives execve and so includes the parent's peak
rss = [int(x.split()[1]) / 1024 for x in open("/proc/self/status") if x.startswith("VmHWM")][0]
print(json.dumps({"import": imp, "total": total, "peak_rss_mb": rss}))
"""
)

WORKER = (
    FACTORY
    + r"""
import json, math, os, statistics, sys, time
task, lib, arg = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])

def mem():
    d = {}
    for line in open("/proc/self/status"):
        k, _, v = line.partition(":")
        if k in ("VmRSS", "VmHWM"):
            d[k] = int(v.split()[0]) / 1024
    return d

def timed(fn, x):
    t = time.perf_counter(); r = fn(x); return time.perf_counter() - t, r

out = {}
if task == "version":
    from importlib import metadata
    dist = {"nupunkt-adaptive": "nupunkt", "regex": None}.get(lib, lib)
    out["version"] = metadata.version(dist) if dist else "stdlib re"
    if dist == "nupunkt":
        import nupunkt
        out["version"] = nupunkt.__version__
    out["python"] = sys.version.split()[0]
    if lib.startswith("nupunkt"):
        import nupunkt
        pkg = os.path.dirname(nupunkt.__file__)
        files = [os.path.join(r, f) for r, _, fs in os.walk(pkg) for f in fs
                 if "__pycache__" not in r]
        out["pkg_bytes"] = sum(os.path.getsize(f) for f in files)
        out["model_bytes"] = sum(os.path.getsize(f) for f in files
                                 if os.sep + "models" + os.sep in f
                                 and f.endswith((".gz", ".bin", ".xz", ".json")))
elif task == "rss":
    base = mem()
    fn = make(lib); fn(open(arg["small"], encoding="utf-8").read())
    loaded = mem()
    r, after = [], {"VmRSS": None, "VmHWM": None}
    if arg["big"]:
        r = fn(open(arg["big"], encoding="utf-8").read())
        after = mem()
    out = {"base_rss": base["VmRSS"], "loaded_rss": loaded["VmRSS"],
           "loaded_peak": loaded["VmHWM"], "after_rss": after["VmRSS"],
           "after_peak": after["VmHWM"], "n": len(r)}
elif task == "throughput":
    fn = make(lib); fn(open(arg["small"], encoding="utf-8").read())
    text = open(arg["path"], encoding="utf-8").read()
    first, r = timed(fn, text)
    xs = [timed(fn, text)[0] for _ in range(arg["runs"])]
    out = {"first": first, "warm": statistics.median(xs), "min": min(xs), "all": xs,
           "n": len(r), "chars": len(text), "bytes": len(text.encode())}
    if reset(lib):
        ys = []
        for _ in range(arg["runs"]):
            reset(lib); ys.append(timed(fn, text)[0])
        out["warm_nomemo"] = statistics.median(ys)
elif task == "perdoc":
    fn = make(lib); fn(open(arg["small"], encoding="utf-8").read())
    docs = json.load(open(arg["path"], encoding="utf-8"))
    def run():
        t = time.perf_counter(); n = sum(len(fn(d)) for d in docs)
        return time.perf_counter() - t, n
    first, n = run()
    xs = [run()[0] for _ in range(arg["runs"])]
    out = {"first": first, "warm": statistics.median(xs), "all": xs, "n": n,
           "docs": len(docs), "chars": sum(map(len, docs)),
           "bytes": sum(len(d.encode()) for d in docs)}
    if reset(lib):
        ys = []
        for _ in range(arg["runs"]):
            reset(lib); ys.append(run()[0])
        out["warm_nomemo"] = statistics.median(ys)
elif task == "scaling":
    fn = make(lib); fn(open(arg["small"], encoding="utf-8").read())
    full = open(arg["path"], encoding="utf-8").read()
    pts = []
    for size in arg["sizes"]:
        if pts:  # power-law extrapolation from the previous sizes
            (c1, t1), (c2, t2) = pts[-2] if len(pts) > 1 else pts[-1], pts[-1]
            p = math.log(t2 / t1) / math.log(c2 / c1) if c2 != c1 else 1.0
            if t2 * (size / c2) ** min(max(p, 1.0), 2.5) > arg["budget"]:
                out[str(size)] = None
                break
        text = full[:size]
        reset(lib); fn(text)
        reset(lib); once = timed(fn, text)[0]
        reps = max(1, int(0.2 / max(timed(fn, text)[0], 1e-6)))
        xs = []
        for _ in range(arg["runs"]):
            t = time.perf_counter()
            for _ in range(reps):
                reset(lib); fn(text)
            xs.append((time.perf_counter() - t) / reps)
        out[str(size)] = statistics.median(xs)
        pts.append((size, out[str(size)]))
print(json.dumps(out))
"""
)


def log(*a: object) -> None:
    print(*a, file=sys.stderr, flush=True)


def med(xs: list[float]) -> float:
    return statistics.median(xs)


# --------------------------------------------------------------------------- corpora
def fetch_gutenberg(work: Path, name: str, gid: int) -> dict:
    raw = work / f"pg{gid}.txt"
    if not raw.exists() or b"*** END OF" not in raw.read_bytes():
        url = f"https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
        log(f"downloading {url}")
        raw.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    data = raw.read_bytes()
    lines = data.decode("utf-8-sig").replace("\r\n", "\n").split("\n")
    start = next(i for i, x in enumerate(lines) if x.startswith("*** START OF"))
    end = next(i for i, x in enumerate(lines) if x.startswith("*** END OF"))
    text = "\n".join(lines[start + 1 : end]).strip() + "\n"
    (work / f"{name}.txt").write_text(text, encoding="utf-8")
    return {"id": gid, "raw_sha256": hashlib.sha256(data).hexdigest(), "chars": len(text)}


def build_corpora(work: Path, repo: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    meta: dict = {}
    for name, gid in GUTENBERG.items():
        try:
            meta[name] = fetch_gutenberg(work, name, gid)
        except Exception as e:  # offline: skip the Gutenberg corpora
            log(f"skipping {name}: {e}")
    docs = []
    with gzip.open(repo / "data" / "test.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line)["text"].replace("<|sentence|>", ""))
    (work / "docs.json").write_text(json.dumps(docs), encoding="utf-8")
    shuffled = docs[:]
    random.Random(1234).shuffle(shuffled)
    out, size = [], 0
    for d in shuffled:
        out.append(d)
        size += len(d) + 2
        if size >= 4_000_000:
            break
    legal = "\n\n".join(out)
    (work / "legal_4mb.txt").write_text(legal, encoding="utf-8")
    alltext = "\n\n".join(shuffled)
    (work / "legal_all.txt").write_text(alltext, encoding="utf-8")
    big = "\n\n".join([alltext] * (50_000_000 // len(alltext) + 1))
    big = big[: big.rfind("\n\n", 0, 50_000_000)]
    (work / "legal_50mb.txt").write_text(big, encoding="utf-8")
    five = big[: big.rfind("\n\n", 0, 5_000_000)]
    (work / "legal_5mb.txt").write_text(five, encoding="utf-8")
    small = legal[:1024]
    (work / "small_1kb.txt").write_text(small[: small.rfind(" ")], encoding="utf-8")
    meta.update(
        legal_4mb={"chars": len(legal), "docs": len(out)},
        legal_all={"chars": len(alltext), "docs": len(docs)},
        legal_50mb={"chars": len(big)},
        legal_5mb={"chars": len(five)},
        docs={"docs": len(docs), "chars": sum(map(len, docs))},
    )
    return meta


# --------------------------------------------------------------------------- envs
def build_envs(work: Path, py: str, wheel: str | None) -> dict[str, str]:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit("--with-comparators needs `uv` on PATH")
    specs = {
        "nupunkt-0.7.0": [wheel or "nupunkt==0.7.0"],
        "nupunkt-0.6.0": ["nupunkt==0.6.0"],
        "nupunkt-0.5.1": ["nupunkt==0.5.1"],
        "nltk": ["nltk"],
        "pysbd": ["pysbd"],
        "sentencex": ["sentencex"],
        "blingfire": ["blingfire", "numpy"],  # blingfire imports numpy undeclared
        "spacy": ["spacy"],
        "syntok": ["syntok"],
    }
    envs = {}
    for name, pkgs in specs.items():
        venv = work / "venvs" / name
        exe = venv / "bin" / "python"
        if not exe.exists():
            log(f"creating venv {name}")
            subprocess.run([uv, "venv", "-q", "-p", py, str(venv)], check=True)
            r = subprocess.run(
                [uv, "pip", "install", "-q", "-p", str(exe), *pkgs], capture_output=True, text=True
            )
            if r.returncode:
                log(f"install failed for {name}; skipping:\n{r.stderr[-500:]}")
                shutil.rmtree(venv)
                continue
            if name == "nltk":
                # nltk >= 3.9.3 refuses data dirs under world-writable parents (e.g. /tmp),
                # so let it pick its default location (usually ~/nltk_data).
                code = "import nltk; nltk.download('punkt_tab', quiet=True)"
                subprocess.run([str(exe), "-W", "ignore", "-c", code], check=False)
        envs[name] = str(exe)
    return envs


def site_packages_mb(exe: str) -> float | None:
    venv = Path(exe).parents[1]
    sp = list(venv.glob("lib/python*/site-packages"))
    if not sp:
        return None
    total = sum(p.stat().st_size for p in sp[0].rglob("*") if p.is_file())
    return total / 1e6


# --------------------------------------------------------------------------- runner
class Runner:
    def __init__(self, cpu: int, work: Path):
        self.pin = ["taskset", "-c", str(cpu)] if cpu >= 0 and shutil.which("taskset") else []
        self.env = {**os.environ, "PYTHONWARNINGS": "ignore"}
        self.env.pop("PYTHONPATH", None)
        self.cwd = str(work)

    def run(self, exe: str, code: str, *args: str, timeout: float | None = None) -> dict:
        cmd = [*self.pin, exe, "-c", code, *args]
        r = subprocess.run(
            cmd, capture_output=True, text=True, env=self.env, cwd=self.cwd, timeout=timeout
        )
        if r.returncode:
            raise RuntimeError(r.stderr[-800:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def wall(self, exe: str, code: str, *args: str) -> float:
        t = time.perf_counter()
        subprocess.run(
            [*self.pin, exe, "-c", code, *args],
            env=self.env,
            cwd=self.cwd,
            check=True,
            capture_output=True,
        )
        return time.perf_counter() - t

    def worker(
        self, exe: str, task: str, lib: str, timeout: float | None = None, **arg: object
    ) -> dict:
        return self.run(exe, WORKER, task, lib, json.dumps(arg), timeout=timeout)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--with-comparators", action="store_true")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cpu", type=int, default=2, help="core for taskset (-1 disables pinning)")
    ap.add_argument("--work-dir", default=None, help="corpora/venvs cache (default: $TMPDIR)")
    ap.add_argument("--python", default="3.13", help="python for comparator venvs")
    ap.add_argument("--wheel", default=None, help="nupunkt 0.7.0 wheel (default: dist/*.whl)")
    ap.add_argument(
        "--max-run-seconds",
        type=float,
        default=60.0,
        help="skip a (library, corpus) pair if one call is estimated slower",
    )
    ap.add_argument("--only", default="", help="comma list of libraries to run")
    ap.add_argument("--json", default=None, help="write raw results here (after each library)")
    ap.add_argument("--resume", action="store_true", help="reuse libraries already in --json")
    a = ap.parse_args()

    tmp = os.environ.get("TMPDIR", "/tmp")
    work = Path(a.work_dir or Path(tmp) / "nupunkt-perf-bench").resolve()
    meta = build_corpora(work, REPO)
    runner = Runner(a.cpu, work)
    R = a.runs

    # (label, python executable, factory name)
    libs: list[tuple[str, str, str]] = []
    if a.with_comparators:
        wheel = a.wheel or next((str(p) for p in sorted((REPO / "dist").glob("*.whl"))), None)
        envs = build_envs(work, a.python, wheel)
        for v in ("0.7.0", "0.6.0", "0.5.1"):
            if f"nupunkt-{v}" in envs:
                libs.append((f"nupunkt {v}", envs[f"nupunkt-{v}"], "nupunkt"))
                if v != "0.5.1":  # 0.5.1 has no adaptive mode
                    libs.append((f"nupunkt {v} adaptive", envs[f"nupunkt-{v}"], "nupunkt-adaptive"))
        for n in ("nltk", "pysbd", "sentencex", "blingfire", "spacy", "syntok"):
            if n in envs:
                libs.append((n, envs[n], n))
    else:
        libs += [
            ("nupunkt (current)", sys.executable, "nupunkt"),
            ("nupunkt (current) adaptive", sys.executable, "nupunkt-adaptive"),
        ]
    libs.append(("regex split (floor)", sys.executable, "regex"))
    if a.only:
        keep = a.only.split(",")
        libs = [x for x in libs if any(k == x[0] or k == x[2] for k in keep)]

    res: dict = {"meta": meta, "loadavg_start": os.getloadavg(), "libs": {}}
    res["machine"] = {
        "cpu": next(
            (
                ln.split(":", 1)[1].strip()
                for ln in Path("/proc/cpuinfo").read_text().splitlines()
                if ln.startswith("model name")
            ),
            platform.processor(),
        ),
        "platform": platform.platform(),
        "pinned": " ".join(runner.pin) or "no",
    }
    small = str(work / "small_1kb.txt")
    bare = [runner.wall(sys.executable, "pass") for _ in range(R)]
    res["bare_interpreter_s"] = med(bare)

    def bench(label: str, exe: str, lib: str) -> dict:
        log(f"== {label}")
        r: dict = runner.worker(exe, "version", lib)
        r["loadavg"] = os.getloadavg()
        if not lib.endswith("adaptive") and lib != "regex":
            r["site_packages_mb"] = site_packages_mb(exe) if exe != sys.executable else None
        # 1. cold start: fresh interpreter, import + first tokenize of 1 KB
        runner.run(exe, COLD, lib, small)  # untimed: populate .pyc / page cache
        colds, walls = [], []
        for _ in range(R):
            t = time.perf_counter()
            colds.append(runner.run(exe, COLD, lib, small))
            walls.append(time.perf_counter() - t)
        imps = [c["import"] for c in colds if c["import"] is not None]
        r["cold"] = {
            "import": med(imps) if imps else None,
            "import_first": med([c["total"] for c in colds]),
            "process_wall": med(walls),
            "peak_rss_mb": med([c["peak_rss_mb"] for c in colds]),
        }
        # 2. throughput, smallest corpus first; later corpora are skipped when a single
        # call is estimated (power-law fit of the last two points) to exceed the budget.
        r["throughput"] = {}
        pts: list[tuple[int, float]] = []

        def estimate(chars: int, pts: list[tuple[int, float]] = pts) -> float:
            if not pts:
                return 0.0
            (c1, t1), (c2, t2) = pts[-2] if len(pts) > 1 else pts[-1], pts[-1]
            p = math.log(t2 / t1) / math.log(c2 / c1) if c2 != c1 else 1.0
            return t2 * (chars / c2) ** min(max(p, 1.0), 2.5)

        for name in ("alice", "sherlock", "legal_4mb", "legal_all", "legal_50mb"):
            p = work / f"{name}.txt"
            if not p.exists():
                continue
            est = estimate(meta[name]["chars"])
            if est > a.max_run_seconds:
                r["throughput"][name] = {"skipped": f"est. {est:.0f} s per call"}
                continue
            try:  # hard cap in case the extrapolation was too optimistic
                cap = a.max_run_seconds * (2 * R + 2) + 60
                t = runner.worker(
                    exe, "throughput", lib, path=str(p), small=small, runs=R, timeout=cap
                )
            except subprocess.TimeoutExpired:
                r["throughput"][name] = {"skipped": f"timed out after {cap:.0f} s"}
                pts.append((meta[name]["chars"], cap))
                continue
            r["throughput"][name] = t
            pts.append((t["chars"], t["warm"]))
        # 3. memory: fresh process, after loading, and peak while tokenizing ~5 MB
        big = str(work / "legal_5mb.txt")
        if estimate(meta["legal_5mb"]["chars"]) > a.max_run_seconds:
            big = ""
        rss = [runner.worker(exe, "rss", lib, small=small, big=big) for _ in range(3)]
        r["rss"] = {k: rss[0][k] if rss[0][k] is None else med([x[k] for x in rss]) for k in rss[0]}
        # 4. many small calls (cost estimated linearly from the smallest corpus)
        c0, t0 = pts[0]
        est = meta["docs"]["chars"] / (c0 / t0)
        if est > a.max_run_seconds:
            r["perdoc"] = {"skipped": f"est. {est:.0f} s per pass"}
        else:
            r["perdoc"] = runner.worker(
                exe, "perdoc", lib, path=str(work / "docs.json"), small=small, runs=R
            )
        # 5. scaling on prefixes of the 50 MB corpus
        n50 = meta["legal_50mb"]["chars"]
        sizes = [min(s, n50) for s in (10_000, 100_000, 1_000_000, 10_000_000, 50_000_000)]
        r["scaling"] = runner.worker(
            exe,
            "scaling",
            lib,
            path=str(work / "legal_50mb.txt"),
            small=small,
            sizes=sizes,
            runs=R,
            budget=a.max_run_seconds,
        )
        return r

    res["errors"] = {}
    done = {}
    if a.json and a.resume and Path(a.json).exists():
        done = json.loads(Path(a.json).read_text())["libs"]
    for label, exe, lib in libs:
        try:
            res["libs"][label] = done[label] if label in done else bench(label, exe, lib)
            if a.json:
                res["loadavg_end"] = os.getloadavg()
                Path(a.json).write_text(json.dumps(res, indent=1))
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            log(f"{label} failed: {e}")
            res["errors"][label] = str(e)[-300:]
    # Interleaved A/B of the nupunkt versions on unseen text (legal_all, first call in a
    # fresh process), so that machine-load drift affects every version equally.
    ab_libs = [(k, e, lib) for k, e, lib in libs if lib == "nupunkt" and not k.endswith("adaptive")]
    if len(ab_libs) > 1:
        log("== interleaved nupunkt A/B")
        path = str(work / "legal_all.txt")
        ab: dict[str, list[float]] = {k: [] for k, _, _ in ab_libs}
        for _ in range(R):
            for k, exe, lib in ab_libs:
                t = runner.worker(exe, "throughput", lib, path=path, small=small, runs=1)
                ab[k].append(t["chars"] / t["first"] / 1e6)
        res["ab_legal_all_first_mchar_s"] = {k: med(v) for k, v in ab.items()}
    res["loadavg_end"] = os.getloadavg()
    report(res)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))


def report(res: dict) -> None:
    L = res["libs"]
    for k, e in res.get("errors", {}).items():
        print(f"ERROR {k}: {e}")
    m = res["machine"]
    print(
        f"# nupunkt performance benchmark\n\n{m['cpu']} | {m['platform']} | pinned: {m['pinned']}"
    )
    print(
        f"load average start {res['loadavg_start']} end {res['loadavg_end']}; "
        f"bare interpreter start {res['bare_interpreter_s'] * 1e3:.1f} ms\n"
    )

    def ms(x: float | None) -> str:
        return "-" if x is None else f"{x * 1e3:,.1f}"

    print("## Cold start and footprint\n")
    print(
        "| library | version | import ms | import+first 1KB ms | process wall ms | "
        "RSS loaded MB | peak RSS 5MB MB | model MB | site-packages MB |"
    )
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for k, r in L.items():
        c, s = r["cold"], r["rss"]
        mb = f"{r['model_bytes'] / 1e6:.3f}" if r.get("model_bytes") is not None else "-"
        sp = f"{r['site_packages_mb']:.1f}" if r.get("site_packages_mb") else "-"
        peak = "skipped" if s["after_peak"] is None else f"{s['after_peak']:.0f}"
        print(
            f"| {k} | {r['version']} | {ms(c['import'])} | {ms(c['import_first'])} | "
            f"{ms(c['process_wall'])} | {s['loaded_rss']:.0f} | {peak} | {mb} | {sp} |"
        )

    names = [
        n for n in ("alice", "sherlock", "legal_4mb", "legal_all", "legal_50mb") if n in res["meta"]
    ]
    for unit, key in (("Mchar/s", "chars"), ("MB/s (UTF-8)", "bytes")):
        print(f"\n## Throughput in {unit}: warm median / warm with memo cleared / first call\n")
        print(
            "| library | "
            + " | ".join(f"{n} ({res['meta'][n]['chars'] / 1e6:.2f}M chars)" for n in names)
            + " | per-doc, 38.5k calls |"
        )
        print("|---|" + "--:|" * (len(names) + 1))
        for k, r in L.items():
            cells = []
            for n in [*names, "perdoc"]:
                t = r["perdoc"] if n == "perdoc" else r["throughput"].get(n, {"skipped": "n/a"})
                if "skipped" in t:
                    cells.append(f"skipped ({t['skipped']})")
                    continue
                v = [t["warm"], t.get("warm_nomemo"), t["first"]]
                cells.append(" / ".join("-" if x is None else f"{t[key] / x / 1e6:.2f}" for x in v))
            print(f"| {k} | " + " | ".join(cells) + " |")

    if "ab_legal_all_first_mchar_s" in res:
        print("\n## Interleaved nupunkt versions, legal_all first call (Mchar/s, median)\n")
        for k, v in res["ab_legal_all_first_mchar_s"].items():
            print(f"- {k}: {v:.2f}")
    print("\n## Scaling on legal_50mb prefixes (ms per call; ns/char in brackets)\n")
    sizes = sorted({int(s) for r in L.values() for s in r["scaling"]})
    print("| library | " + " | ".join(f"{s / 1e6:g}M" for s in sizes) + " |")
    print("|---|" + "--:|" * len(sizes))
    for k, r in L.items():
        sc = r["scaling"]
        print(
            f"| {k} | "
            + " | ".join(
                f"{sc[str(s)] * 1e3:,.2f} [{sc[str(s)] / s * 1e9:.0f}]" if sc.get(str(s)) else "-"
                for s in sizes
            )
            + " |"
        )


if __name__ == "__main__":
    main()
