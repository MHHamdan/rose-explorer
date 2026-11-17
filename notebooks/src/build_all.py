"""Build every chapter notebook and notebooks/README.md from the chapter modules in this directory.

python notebooks/src/build_all.py            # write notebooks/*.ipynb and notebooks/README.md
Chapters are the modules ch??_*.py; each exposes build() -> Chapter (built, written to notebooks/).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nbkit  # noqa: E402

ROOT = nbkit.ROOT


def chapter_modules():
    return sorted(p.stem for p in HERE.glob("ch[0-9][0-9]_*.py"))


def build_all():
    chapters = []
    for name in chapter_modules():
        mod = importlib.import_module(name)
        chapters.append(mod.build())
    write_readme(chapters)
    return chapters


def write_readme(chapters):
    reg = nbkit.REGISTRY
    covered = {i for c in chapters for i in c.ids}
    rows = []
    for c in chapters:
        colab = nbkit.COLAB_URL.format(slug=nbkit.GITHUB_SLUG, name=c.filename)
        rows.append(f"| {c.number} | [{c.title}]({c.filename}) | {c.n_eq} | {len(set(c.ids))} | "
                    f"[![Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab}) |")
    missing = sorted(set(reg) - covered)
    text = f"""# The mathematics of the survey, one chapter per topic

These notebooks explain and verify every equation of the paper and its supplement: {len(covered)} of the
{len(reg)} entries in [`data/equation_registry.json`](../data/equation_registry.json). Each equation is given
three times: **in plain words** for a first-time learner, as a **refresher** for an advanced reader
(assumptions, derivation sketch, where it breaks), and as **code that verifies it**. Running a notebook end to
end is the verification: its last cell fails if any check failed.

Prefer to start by turning knobs? The repository's interactive explorer (`app/`, also on the GitHub Pages site)
animates the same results: each tab names the chapter that verifies it.

The notebooks are self-contained (numpy, scipy, sympy, matplotlib, torch on CPU) and run on Google Colab
without this repository. Equation ids such as `EQ-M-007` name the entry in the registry, which maps it to a
line in the paper source, so a reader can check the notebook against the paper and the paper against the
notebook.

| # | Chapter | Entries | Registry ids | Open |
|---|---|---|---|---|
{chr(10).join(rows)}

{"All registry entries are covered." if not missing else "Not yet covered: " + ", ".join(f"`{m}`" for m in missing) + "."}

## How the chapters are built and checked

The `.ipynb` files are generated from the Python modules in [`src/`](src/) (one per chapter) by
`python notebooks/src/build_all.py`; edit the module, not the notebook. `python tools/check_notebooks.py`
rebuilds every chapter and fails if a committed notebook differs, executes each notebook on the CPU and fails
on any error or failed check, and reports registry entries that no chapter covers. Run it before committing; continuous
integration runs it on every push.

## Contributing

Corrections and clearer explanations are welcome; they are judged by whether they make an equation more
accurate or easier to understand and verify.

1. Find the entry by its equation id (search the chapter module for `EQ-M-...`).
2. Edit the plain-words text, the refresher, or the verification code in `notebooks/src/chNN_*.py`. A
   verification must call `check(name, ok, detail)` and must run on a CPU in seconds.
3. Run `python tools/check_notebooks.py` and commit the module together with the regenerated notebook.
4. If you believe the **paper** is wrong, say so in the issue with the equation id and the failing check;
   the registry entry's `verified` field and the paper text are then corrected together.

The Colab links use the repository slug `{nbkit.GITHUB_SLUG}`, filled in when the repository is published.
"""
    (ROOT / "notebooks" / "README.md").write_text(text, encoding="utf-8")
    (ROOT / "notebooks" / "coverage.json").write_text(json.dumps(
        {"chapters": {c.filename: sorted(set(c.ids)) for c in chapters}, "uncovered": missing}, indent=1) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    ch = build_all()
    print(f"built {len(ch)} chapters, {sum(c.n_eq for c in ch)} entries, "
          f"{len({i for c in ch for i in c.ids})}/{len(nbkit.REGISTRY)} registry ids covered")
