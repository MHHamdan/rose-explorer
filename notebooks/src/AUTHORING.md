# Writing a chapter

A chapter is a module `chNN_<slug>.py` in this directory:

```python
from nbkit import Chapter

def build():
    c = Chapter(3, "variational_inference_vaes", "Variational inference and VAEs",
                summary="...two or three sentences on what the chapter covers and why it matters...",
                prerequisites="Chapters 1 and 2", paper_sections="main §3.2–3.3; Supplement S1.5, S4.2")
    c.section("Why bound the likelihood at all?", "...a short orientation paragraph...")
    c.eq(["EQ-M-007", "EQ-S1-014"], "The evidence lower bound", r"\log p_\theta(\mathbf{x}) \ge ...",
         plain="...", refresher="...", code="...")
    ...
    return c.build()          # returns the Chapter; the notebook is at c.path

if __name__ == "__main__":
    print(build().path)
```

## Rules

**Coverage.** Every registry id assigned to the chapter appears in exactly one `c.eq(...)` call. Equivalent
statements from the main paper and the supplement share one entry (main id first). Prose-only entries
(`(prose) ...`, `(arithmetic) ...`, table cells) still get an entry: state the claim as an equation or an
inequality and verify it on a concrete instance.

**LaTeX.** Plain LaTeX only, no paper macros: write `D_{\mathrm{KL}}(p\,\|\,q)`, `\mathbb{E}`, `\mathbf{x}`,
`\bar\alpha_t`, `\boldsymbol{\epsilon}_\theta`, `p_{\mathrm{data}}`. Display math is passed without `$$`.
Inline math in the Markdown uses `$...$`.

**In plain words** (first-time learner). Name every symbol the first time it appears, say what the equation
claims in one or two sentences, and give the intuition or a tiny concrete example. No derivation, no jargon
that the chapter has not introduced.

**Refresher** (advanced reader). The assumptions (support, integrability, differentiability, population vs
finite sample), a derivation sketch in three to six steps, what changes when an assumption fails, and how the
equation is used in the paper (which experiment or claim relies on it). Cite the original source by author and
year in prose when the paper does.

**Verify.** A code cell that runs on a CPU in a few seconds, uses only numpy/scipy/sympy/matplotlib/torch
(already imported by the setup cell), and reports through `check(name, ok, detail)`; `close(a, b, tol, what)`
returns `(ok, message)` for numeric comparisons. Prefer an exact check (closed form, symbolic identity, small
discrete distribution where sums are exact) over a Monte-Carlo one; if Monte Carlo is the only option, state
the standard error in `detail` and use a tolerance of several standard errors. Where
`verification/verify_equations.py` already has a check for the id (the registry field
`check_in_verify_equations`), port its logic so the two agree; checks named `repo_*` depend on this
repository's model code and are not ported (say in the refresher that the repository check covers them). A
figure is welcome when it shows the mechanism (mode covering vs seeking, the DDIM family, the FID bias curve);
keep it small and close it with `plt.show()`.

**Honesty.** If a check needs a tolerance that is loose, say why. If the paper's statement holds only under a
condition the paper does not state, say so in the refresher and open an issue against the paper. Never weaken
a check to make it pass.

**Style.** British spelling, as in the paper. Sentences, not fragments. No marketing language.

## Building and checking

```bash
python notebooks/src/ch03_variational_inference_vaes.py      # writes notebooks/03_variational_inference_vaes.ipynb
python tools/check_notebooks.py --only 03                     # rebuild + execute this chapter
python notebooks/src/build_all.py                             # all chapters + README.md + coverage.json
python tools/check_notebooks.py                               # the full gate (also run by CI)
```
