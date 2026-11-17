"""Render the explorer's tabs to site/assets/*.png with captions, without a browser.

Each preview is the demo's own figure at its default settings plus the first line of its explanation, so the
landing page shows what the app shows. Run after changing app/genai_explorer/demos.py:

python site/make_previews.py      # writes site/assets/preview_*.png and site/assets/previews.json
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
from genai_explorer import demos  # noqa: E402

OUT = ROOT / "site" / "assets"


def strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = text.replace("D_{\\mathrm{KL}}(p\\|q)", "KL(p‖q)").replace("D_{\\mathrm{KL}}(q\\|p)", "KL(q‖p)")
    text = re.sub(r"\$([^$]+)\$", r"\1", text)
    return text.replace("\\beta", "β").replace("\\eta", "η").replace("\\bar\\alpha", "ᾱ")


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for i, (name, fn, chapter, lead) in enumerate(demos.TABS):
        fig, md = fn()
        path = OUT / f"preview_{i + 1}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight", facecolor="white", metadata={"Software": None})
        first = strip_md(md.split("\n\n")[0])
        rows.append({"tab": name, "lead": lead, "chapter": chapter, "image": path.name, "caption": first})
        print(f"{name:26s} -> {path.name}")
    (OUT / "previews.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
