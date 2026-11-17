"""Gate for the equation notebooks (notebooks/*.ipynb).

1. Rebuilds every chapter from notebooks/src/ into a temporary directory and fails if a committed notebook or
   notebooks/README.md differs (the notebooks are generated files).
2. Executes every notebook on the CPU with a fresh kernel and fails on any cell error; the last cell of each
   notebook asserts that all of its checks passed.
3. Reports registry entries (data/equation_registry.json) that no chapter covers; with --strict, fails on them.

python tools/check_notebooks.py            # all three
python tools/check_notebooks.py --no-exec  # 1 and 3 only (seconds)
python tools/check_notebooks.py --only 03  # execute chapters whose file name starts with 03
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
SRC = NB / "src"
sys.path.insert(0, str(SRC))


def rebuild_and_compare():
    import nbkit
    import build_all
    bad = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        nbkit.OUT = tmp                       # Chapter.build() writes here
        build_all.ROOT = ROOT
        chapters = []
        for name in build_all.chapter_modules():
            mod = __import__(name)
            chapters.append(mod.build())
        # README/coverage are written under ROOT/notebooks by write_readme; redirect by comparing text instead
        readme_committed = (NB / "README.md").read_text(encoding="utf-8") if (NB / "README.md").exists() else ""
        cov_committed = (NB / "coverage.json").read_text(encoding="utf-8") if (NB / "coverage.json").exists() else ""
        build_all.write_readme(chapters)
        readme_new = (NB / "README.md").read_text(encoding="utf-8")
        cov_new = (NB / "coverage.json").read_text(encoding="utf-8")
        (NB / "README.md").write_text(readme_committed, encoding="utf-8")
        (NB / "coverage.json").write_text(cov_committed, encoding="utf-8")
        if readme_new != readme_committed:
            bad.append("notebooks/README.md")
        if cov_new != cov_committed:
            bad.append("notebooks/coverage.json")
        for c in chapters:
            committed = NB / c.filename
            if not committed.exists() or not filecmp.cmp(tmp / c.filename, committed, shallow=False):
                bad.append(f"notebooks/{c.filename}")
        for p in NB.glob("*.ipynb"):
            if not (tmp / p.name).exists():
                bad.append(f"notebooks/{p.name} has no source module")
    return chapters, bad


def execute(path: Path, timeout: int) -> tuple[bool, str, float]:
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(path, as_version=4)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    t0 = time.time()
    try:
        NotebookClient(nb, timeout=timeout, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}},
                       env=env).execute()
    except Exception as ex:  # noqa: BLE001
        msg = str(ex)
        return False, msg[-1500:], time.time() - t0
    return True, "", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--strict", action="store_true", help="fail on uncovered registry entries")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    fail = []
    chapters, stale = rebuild_and_compare()
    for s in stale:
        print(f"  FAIL stale or orphaned: {s} (run python notebooks/src/build_all.py)")
    fail += stale
    if not a.no_exec:
        for c in chapters:
            if a.only and not c.filename.startswith(a.only):
                continue
            ok, msg, dt = execute(NB / c.filename, a.timeout)
            print(f"  {'ok  ' if ok else 'FAIL'} {c.filename} ({dt:.0f}s)" + ("" if ok else f"\n{msg}"))
            if not ok:
                fail.append(c.filename)
    reg = json.loads((ROOT / "data" / "equation_registry.json").read_text(encoding="utf-8"))
    covered = {i for c in chapters for i in c.ids}
    missing = sorted(e["id"] for e in reg if e["id"] not in covered)
    dup = {}
    for c in chapters:
        for i in set(c.ids):
            dup.setdefault(i, []).append(c.filename)
    dups = {k: v for k, v in dup.items() if len(v) > 1}
    print(f"  coverage: {len(covered)}/{len(reg)} registry entries in {len(chapters)} chapters"
          + (f"; uncovered: {', '.join(missing)}" if missing else "")
          + (f"; in several chapters: {dups}" if dups else ""))
    if a.strict and missing:
        fail.append("coverage")
    print(("[FAIL] notebooks: " if fail else "[PASS] notebooks: ") + f"{len(chapters)} chapters, {len(fail)} failures")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
