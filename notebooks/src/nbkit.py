"""Shared builder for the equation notebooks in notebooks/*.ipynb.

Each chapter is a Python module notebooks/src/chNN_<slug>.py that creates a `Chapter`, adds one entry per
paper equation with `Chapter.eq(...)`, and calls `Chapter.build()`. The .ipynb files are generated, never
edited by hand (tools/check_notebooks.py rebuilds and compares them, then executes them on the CPU).

Every entry has three parts in a fixed order so that a reader can stop at the depth they want:

  * In plain words  -- for a first-time learner: what the symbols are and what the equation says.
  * Refresher       -- for an advanced reader: assumptions, a derivation sketch, where it breaks.
  * Verify          -- a code cell that checks the identity numerically or symbolically and calls
                       `check(name, ok, detail)`; the notebook's last cell fails if any check failed.

Notebooks are self-contained (numpy, scipy, sympy, matplotlib, torch on CPU) so that they run on Colab
without this repository. The equation ids (EQ-M-..., EQ-S1-...) and the paper locations come from
data/equation_registry.json at build time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import nbformat
from learning import LEARNING
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "notebooks"
REGISTRY = {e["id"]: e for e in json.loads((ROOT / "data" / "equation_registry.json").read_text(encoding="utf-8"))}

# Filled in when the repository is published; tools/repository_integrity_check.py lists it as a placeholder.
GITHUB_SLUG = "MHHamdan/rose-explorer"
COLAB_URL = "https://colab.research.google.com/github/{slug}/blob/main/notebooks/{name}"

# chapter -> explorer tab that illustrates it (app/genai_explorer/demos.py TABS)
EXPLORER_TABS = {1: "Divergences", 2: "Divergences", 4: "Language-model decoding", 6: "GAN critics",
                 7: "Score and Langevin", 8: "Diffusion sampler", 9: "Diffusion sampler", 10: "Evaluation metrics",
                 11: "Post-training tilt"}

NOTATION = r"""
| Symbol | Meaning |
|---|---|
| $\mathbf{x}$, $\mathbf{z}$ | an observed data point; a latent (unobserved) variable |
| $p_{\mathrm{data}}$, $p_\theta$ | the data distribution; the model's distribution with parameters $\theta$ |
| $\mathbb{E}_p[f]$ | the average of $f$ when its argument is drawn from $p$ |
| $D_{\mathrm{KL}}(p\,\Vert\,q)$ | Kullback--Leibler divergence, $\mathbb{E}_p[\log p - \log q]$ |
| $\mathcal{N}(\mu,\Sigma)$ | a Gaussian with mean $\mu$ and covariance $\Sigma$ |
| $\epsilon$ | standard Gaussian noise, $\epsilon\sim\mathcal{N}(0,I)$ |
| $\bar\alpha_t$, $\beta_t$ | diffusion noise-schedule quantities, $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$ |
| $\mathbf{s}_\theta(\mathbf{x})$ | a score network, an estimate of $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ |
| $\mathbf{c}$ | a conditioning signal (a class label, a text prompt) |
"""

SETUP_CODE = '''# Setup: CPU only, small sizes, fixed seeds. Everything here is pre-installed on Google Colab.
import math, os, sys, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the checks are meant to run on a CPU
import numpy as np
import scipy, scipy.stats, scipy.integrate, scipy.linalg, scipy.optimize, scipy.special
import sympy as sp
import matplotlib
import matplotlib.pyplot as plt
import torch
torch.set_num_threads(2)
rng = np.random.default_rng(0)
torch.manual_seed(0)

RESULTS = []   # (name, passed, detail) for every check in this notebook

def check(name, ok, detail=""):
    """Record one verification result and print it. The last cell fails if any check failed."""
    ok = bool(ok)
    RESULTS.append((name, ok, str(detail)))
    print(("PASS  " if ok else "FAIL  ") + name + ("  |  " + str(detail) if detail else ""))
    return ok

def close(a, b, tol, what=""):
    """|a - b| <= tol elementwise; returns (ok, message) for check()."""
    err = float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))
    return err <= tol, f"{what}|diff| = {err:.3e} (tolerance {tol:.1e})"

print("numpy", np.__version__, "| scipy", scipy.__version__, "| sympy", sp.__version__, "| torch", torch.__version__)
'''

SUMMARY_CODE = '''# Summary of every check in this notebook. This cell raises if any check failed, so that running the
# notebook end to end (locally, on Colab, or in the repository's gate) is itself the verification.
n_ok = sum(ok for _, ok, _ in RESULTS)
print(f"{n_ok}/{len(RESULTS)} checks passed")
for name, ok, detail in RESULTS:
    print(("  ok    " if ok else "  FAIL  ") + name)
failed = [name for name, ok, _ in RESULTS if not ok]
assert not failed, f"failed checks: {failed}"
'''


class Chapter:
    def __init__(self, number: int, slug: str, title: str, summary: str, prerequisites: str = "",
                 paper_sections: str = ""):
        self.number, self.slug, self.title = number, slug, title
        self.summary, self.prerequisites, self.paper_sections = summary, prerequisites, paper_sections
        self.cells = []
        self.ids: list[str] = []
        self.titles: list[str] = []
        self.n_eq = 0
        self.filename = f"{number:02d}_{slug}.ipynb"

    # ---- content -------------------------------------------------------------------------------------------
    def md(self, text: str):
        self.cells.append(new_markdown_cell(text.strip("\n")))

    def code(self, src: str):
        self.cells.append(new_code_cell(src.strip("\n")))

    def section(self, title: str, text: str = ""):
        self.md(f"## {title}\n\n{text}" if text else f"## {title}")

    def eq(self, ids, title: str, latex: str, plain: str, refresher: str, code: str, *, note: str = ""):
        """One paper equation (or a group of equivalent statements) with its three layers.

        ids: registry ids this entry covers (the main-paper id first, then supplement duplicates).
        latex: display math WITHOUT surrounding $$ (plain LaTeX; the paper's macros expanded).
        plain / refresher: Markdown. code: Python using check()/close() from the setup cell.
        """
        ids = [ids] if isinstance(ids, str) else list(ids)
        for i in ids:
            if i not in REGISTRY:
                raise KeyError(f"{i} is not in data/equation_registry.json")
        self.ids += ids
        self.n_eq += 1
        self.titles.append(title)
        where = "; ".join(f"`{i}` {_where(REGISTRY[i])}" for i in ids)
        head = (f"### {self.number}.{self.n_eq} {title}\n\n"
                f"*Paper:* {where}\n\n"
                f"$$\n{latex.strip()}\n$$\n")
        if note:
            head += f"\n> {note.strip()}\n"
        self.md(head)
        self.md(f"**In plain words.** {plain.strip()}\n\n**Refresher.** {refresher.strip()}")
        self.md("**Verify.**")
        self.code(code)

    # ---- output ----------------------------------------------------------------------------------------------
    def build(self):
        nb = new_notebook()
        badge = (f"[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                 f"({COLAB_URL.format(slug=GITHUB_SLUG, name=self.filename)})")
        intro = (f"# Chapter {self.number}: {self.title}\n\n{badge}\n\n{self.summary.strip()}\n\n"
                 f"**Paper sections:** {self.paper_sections or 'see each entry'}.  "
                 f"**Prerequisites:** {self.prerequisites or 'none beyond Chapter 1'}.\n\n"
                 "**How to read this chapter.** Each equation has three layers. *In plain words* is for a first "
                 "encounter. *Refresher* states the assumptions and sketches the derivation for readers who have "
                 "seen the material. *Verify* checks the identity in code; run every cell (Runtime > Run all) and "
                 "the last cell reports whether every check passed. Equation ids such as `EQ-M-007` refer to "
                 "`data/equation_registry.json` in the repository, which maps each id to its line in the paper.\n\n"
                 "**Found an error?** Open an issue quoting the equation id, or fix the entry in "
                 f"`notebooks/src/{Path(self.filename).stem}.py` and run `python tools/check_notebooks.py`.\n\n"
                 f"<details><summary>Notation used throughout</summary>\n{NOTATION}\n</details>")
        nb.cells.append(new_markdown_cell(intro))
        nb.cells.append(new_code_cell(SETUP_CODE.strip("\n")))
        contents = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(self.titles))
        tab = EXPLORER_TABS.get(self.number)
        explore = (f"\n\n**See it move.** The *{tab}* tab of the ROSE explorer (`app/` in the repository, also on the "
                   "project's GitHub Pages site) animates this chapter's main result." if tab else "")
        nb.cells.append(new_markdown_cell(f"## In this chapter\n\n{contents}{explore}"))
        nb.cells += self.cells
        aids = LEARNING.get(self.number)
        if aids:
            take = "\n".join(f"- {t}" for t in aids["takeaways"])
            qs = "\n\n".join(f"**{i + 1}. {q}**\n\n<details><summary>Show the answer</summary>\n\n{a}\n\n</details>"
                               for i, (q, a) in enumerate(aids["questions"]))
            nb.cells.append(new_markdown_cell(f"## Key takeaways\n\n{take}\n\n## Check your understanding\n\n{qs}"))
        nb.cells.append(new_markdown_cell("## Summary of checks"))
        nb.cells.append(new_code_cell(SUMMARY_CODE.strip("\n")))
        for i, cell in enumerate(nb.cells):            # deterministic ids: a rebuild must byte-match the committed file
            cell["id"] = f"ch{self.number:02d}-{i:03d}"
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        nb.metadata["rose"] = {"chapter": self.number, "slug": self.slug, "equation_ids": sorted(set(self.ids))}
        OUT.mkdir(exist_ok=True)
        self.path = OUT / self.filename
        nbformat.write(nb, self.path)
        return self   # the Chapter, so that callers can read .ids / .filename / .path


def _where(e: dict) -> str:
    loc = e["location"]
    if loc.startswith("removed"):
        m = re.search(r"\(EQ-S1-\d+\)", loc)
        return f"removed from the main paper in Oct 2026; now {m.group(0)[1:-1] if m else 'Supplement S1'}"
    f, _, lines = loc.partition(":")
    doc = {"main": "main paper", "S1": "Supplement S1", "S4": "Supplement S4", "S5": "Supplement S5",
           "S10": "Supplement S10"}.get(e["document"], e["document"])
    label = e["label"]
    label = label if label.startswith("(") else f"`{label}`"
    return f"{doc}, `{Path(f).name}:{lines}`, {label}"
