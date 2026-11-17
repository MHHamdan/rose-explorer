"""The interactive explorer (app/) and the landing page (site/) stay consistent with the notebooks."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from genai_explorer import demos, links  # noqa: E402


@pytest.mark.parametrize("name,fn", [(t[0], t[1]) for t in demos.TABS])
def test_every_demo_returns_figure_and_text(name, fn):
    import matplotlib.figure
    fig, md = fn()
    assert isinstance(fig, matplotlib.figure.Figure) and isinstance(md, str) and len(md) > 100, name
    assert "notebook chapter" in md, f"{name}: the text must point to its notebook chapter"


def test_edge_settings_do_not_crash():
    demos.decoding(0.0, 0.1, 1, 3)                  # greedy with maximal truncation
    demos.diffusion(999, 2, 1.0, 5.0, 1)            # fewest steps, stochastic, strong guidance
    demos.tilt(0.05, True, 64)                      # near-zero beta with a wrong reward
    demos.langevin(0.2, 10, 5.0, 2)
    for m in ("drops two of four modes", "blurry (too much spread)", "memorises the training set"):
        demos.metrics(50, m, 1)


def test_demo_numbers_match_the_notebook_claims():
    _, md = demos.tilt(0.5, False, 16)
    assert "0.25 → 0.71" in md                        # the tilt raises the success probability
    _, md = demos.gan_divergences(3.0)
    assert "**JS** = 0.693" in md                     # saturated at log 2 for disjoint supports
    _, md = demos.diffusion(300, 20, 0.0, 0.0, 0)
    assert "modes covered: **8/8**" in md


def test_chapter_links_match_the_notebooks():
    cov = json.loads((ROOT / "notebooks" / "coverage.json").read_text())["chapters"]
    assert sorted(cov) == sorted(f for _, f, _, _ in links.CHAPTERS)
    for n, fname, _, _ in links.CHAPTERS:
        assert fname.startswith(f"{n:02d}_") and (ROOT / "notebooks" / fname).exists()
    for _, _, ch, _ in demos.TABS:
        links.chapter(ch)


def test_ui_builds():
    gr = pytest.importorskip("gradio")
    from genai_explorer.ui import build
    demo = build()
    assert isinstance(demo, gr.Blocks)


def test_site_is_current():
    r = subprocess.run([sys.executable, "site/build_site.py", "--check"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
