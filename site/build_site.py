"""Build the GitHub Pages landing page, site/index.html.

The page embeds the Gradio app of app/app.py from a running copy (links.SPACE_URL, a Hugging Face Space) and,
until one is deployed, shows a gallery of the app's tabs rendered by site/make_previews.py from the same demo
code. The chapter cards come from app/genai_explorer/links.py.

Why not run the app in the browser? Gradio-Lite (Pyodide) was tried on 2026-11-12 and fails upstream for every
published version: the current anyio release requires typing_extensions>=4.16 while Pyodide 0.27's micropip keeps
its bundled 4.12 and ignores overrides from <gradio-requirements>, so `import gradio` raises before any app code
runs (5.45 fails earlier, on an unresolvable huggingface-hub<1.0 pin). Revisit when Pyodide or Lite update.

python site/build_site.py            # write site/index.html
python site/build_site.py --check    # exit 1 if the committed page is stale
"""
from __future__ import annotations

import argparse
import html
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "genai_explorer"
OUT = ROOT / "site" / "index.html"

_spec = importlib.util.spec_from_file_location("links", APP / "links.py")
links = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(links)

PREVIEWS = ROOT / "site" / "assets" / "previews.json"

STAGES = [
    ("Route", "How a model gets around the intractable likelihood: compute it exactly, bound it, or avoid it.",
     "autoregressive · flows · VAEs · diffusion · GANs"),
    ("Objective", "The route fixes the loss and the divergence it minimises, and so which data the model covers.",
     "forward KL · reverse KL · Jensen-Shannon · Wasserstein · Fisher"),
    ("Sampler", "The same trained weights can be sampled many ways; the sampler sets quality, diversity and cost.",
     "greedy · top-p · Langevin · DDPM · DDIM · guidance"),
    ("Evaluation", "What a model can report decides which metrics are admissible and what they cannot see.",
     "bits/dim · perplexity · FID · precision/recall · human judgments"),
    ("Systems", "Post-training, retrieval and agents recombine the routes; each stage is anchored to the last.",
     "SFT · RLHF · DPO · GRPO · RAG · agents"),
]


def chapter_cards() -> str:
    out = []
    for n, fname, title, summary in links.CHAPTERS:
        out.append(f"""      <article class="card">
        <div class="num">{n:02d}</div>
        <h3>{html.escape(title)}</h3>
        <p>{html.escape(summary)}</p>
        <div class="links"><a class="btn small" href="{links.colab(fname)}" target="_blank" rel="noopener">Open in Colab</a>
          <a class="btn small ghost" href="{links.github(fname)}" target="_blank" rel="noopener">View notebook</a></div>
      </article>""")
    return "\n".join(out)


def stage_cards() -> str:
    return "\n".join(f"""      <li class="stage"><span class="step">{i + 1}</span><h3>{name}</h3><p>{html.escape(text)}</p>
        <p class="tags">{html.escape(tags)}</p></li>""" for i, (name, text, tags) in enumerate(STAGES))


def explorer_block() -> str:
    """The live app in an iframe when a Space URL is set; otherwise the preview gallery."""
    if links.SPACE_URL:
        return (f'    <iframe class="app-frame" src="{html.escape(links.SPACE_URL)}" title="ROSE explorer" '
                'allow="clipboard-write" loading="lazy"></iframe>')
    rows = json.loads(PREVIEWS.read_text(encoding="utf-8"))
    cards = []
    for r in rows:
        n, fname, title, _ = links.chapter(r["chapter"])
        cards.append(f"""      <figure class="preview">
        <img src="assets/{r['image']}" alt="{html.escape(r['tab'])}: {html.escape(r['caption'])}" loading="lazy">
        <figcaption><b>{html.escape(r['tab'])}</b> <span class="lead-tag">{html.escape(r['lead'])}</span>
          <p>{html.escape(r['caption'])}</p>
          <a href="{links.colab(fname)}" target="_blank" rel="noopener">Chapter {n}: {html.escape(title)} →</a></figcaption>
      </figure>""")
    return "\n".join(cards)


TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ROSE explorer: generative AI, one knob at a time</title>
<meta name="description" content="Interactive explorer and verified notebooks for the ROSE survey of generative models: routes around the likelihood, objectives, samplers, evaluation and systems.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f8fafc;--panel:#ffffff;--ink:#0f172a;--muted:#475569;--line:#e2e8f0;--accent:#f97316;--accent-ink:#ffffff;
      --accent-soft:#fff1e6;--blue:#2563eb;--blue-soft:#e8efff;--radius:14px;--shadow:0 1px 2px rgba(15,23,42,.06),0 6px 24px rgba(15,23,42,.06)}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b1020;--panel:#121a2e;--ink:#e5e9f2;--muted:#9aa6bd;
      --line:#24304a;--accent:#fb923c;--accent-ink:#111;--accent-soft:#2a1d12;--blue:#7aa2ff;--blue-soft:#17223f;--shadow:none}}
:root[data-theme=dark]{--bg:#0b1020;--panel:#121a2e;--ink:#e5e9f2;--muted:#9aa6bd;--line:#24304a;--accent:#fb923c;
      --accent-ink:#111;--accent-soft:#2a1d12;--blue:#7aa2ff;--blue-soft:#17223f;--shadow:none}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
a{color:var(--blue)}
.wrap{max-width:1180px;margin:0 auto;padding:0 20px}
header.top{position:sticky;top:0;z-index:20;background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--line)}
header.top .wrap{display:flex;align-items:center;gap:18px;height:58px}
.logo{font-weight:700;letter-spacing:.2px;text-decoration:none;color:var(--ink);display:flex;align-items:center;gap:8px}
.logo .dot{width:12px;height:12px;border-radius:4px;background:linear-gradient(135deg,var(--accent),var(--blue))}
nav{display:flex;gap:16px;margin-left:auto;flex-wrap:wrap}
nav a{color:var(--muted);text-decoration:none;font-weight:500;font-size:15px}
nav a:hover{color:var(--ink)}
.hero{padding:56px 0 28px}
.eyebrow{display:inline-block;font-weight:600;font-size:13px;color:var(--accent);background:var(--accent-soft);padding:4px 10px;border-radius:999px}
h1{font-size:clamp(30px,4.6vw,50px);line-height:1.1;margin:14px 0 14px;letter-spacing:-.5px}
.lead{font-size:clamp(16px,2vw,19px);color:var(--muted);max-width:820px;margin:0 0 22px}
.btn{display:inline-flex;align-items:center;gap:8px;background:var(--accent);color:var(--accent-ink);border:1px solid var(--accent);
     text-decoration:none;font-weight:600;padding:10px 16px;border-radius:10px;font-size:15px}
.btn.ghost{background:transparent;color:var(--ink);border-color:var(--line)}
.btn.small{padding:6px 11px;font-size:13.5px;border-radius:8px}
.cta{display:flex;gap:10px;flex-wrap:wrap}
.stats{display:flex;gap:28px;flex-wrap:wrap;margin-top:26px;color:var(--muted);font-size:14px}
.stats b{display:block;color:var(--ink);font-size:22px}
section{padding:34px 0}
h2{font-size:clamp(22px,3vw,30px);margin:0 0 6px;letter-spacing:-.3px}
.sub{color:var(--muted);margin:0 0 22px;max-width:820px}
ol.chain{list-style:none;padding:0;margin:0;display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px;box-shadow:var(--shadow);position:relative}
.stage .step{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:8px;background:var(--blue-soft);color:var(--blue);font-weight:700;font-size:13px}
.stage h3{margin:8px 0 4px;font-size:17px}
.stage p{margin:0;color:var(--muted);font-size:14px}
.stage p.tags{margin-top:8px;font-size:12.5px;color:var(--accent)}
.app-shell{background:var(--panel);border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow);overflow:hidden}
.app-bar{display:flex;align-items:center;gap:8px;padding:10px 14px;border-bottom:1px solid var(--line);font-size:13px;color:var(--muted)}
.app-bar i{width:10px;height:10px;border-radius:50%;background:var(--line);display:inline-block}
.app-bar span{margin-left:8px}
.app-frame{display:block;width:100%;height:860px;border:0;background:#fff}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:0;}
.preview{margin:0;padding:16px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.preview img{width:100%;height:auto;border-radius:8px;border:1px solid var(--line);background:#fff}
.preview figcaption{font-size:14px;color:var(--muted);margin-top:10px}
.preview figcaption b{color:var(--ink)}
.preview .lead-tag{display:block;font-size:12.5px;color:var(--accent);margin:2px 0 6px}
.preview p{margin:0 0 6px}
.run{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:14px 16px;border-top:1px solid var(--line);font-size:14px;color:var(--muted)}
.run code{font-size:.95em}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px 16px 14px;box-shadow:var(--shadow);display:flex;flex-direction:column}
.card .num{font-weight:700;color:var(--accent);font-size:13px}
.card h3{margin:4px 0 6px;font-size:16.5px}
.card p{margin:0 0 12px;color:var(--muted);font-size:14px;flex:1}
.card .links{display:flex;gap:8px;flex-wrap:wrap}
.how{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}
.how div{background:var(--panel);border:1px solid var(--line);border-radius:var(--radius);padding:16px}
.how h3{margin:0 0 6px;font-size:16px}
.how p{margin:0;color:var(--muted);font-size:14px}
code{background:var(--blue-soft);padding:1px 6px;border-radius:6px;font-size:.9em}
footer{border-top:1px solid var(--line);padding:22px 0 40px;color:var(--muted);font-size:14px}
.toggle{background:none;border:1px solid var(--line);color:var(--muted);border-radius:8px;padding:4px 8px;cursor:pointer;font:inherit;font-size:13px}
@media (max-width:900px){ol.chain{grid-template-columns:1fr 1fr}.how{grid-template-columns:1fr}nav{display:none}}
@media (max-width:520px){ol.chain{grid-template-columns:1fr}.wrap{padding:0 16px}}
</style>
</head>
<body>
<header class="top"><div class="wrap">
  <a class="logo" href="#"><span class="dot"></span>ROSE explorer</a>
  <nav><a href="#chain">The chain</a><a href="#explorer">Explorer</a><a href="#chapters">Notebooks</a><a href="#verified">How it is verified</a>
    <a href="https://github.com/__SLUG__" target="_blank" rel="noopener">GitHub</a></nav>
  <button class="toggle" id="theme" aria-label="Toggle colour theme">◐</button>
</div></header>

<main>
<section class="hero"><div class="wrap">
  <span class="eyebrow">Generative AI, explained and verified</span>
  <h1>How generative AI works,<br>one knob at a time.</h1>
  <p class="lead">Every generative model, from a VAE to a large language model to a text-to-image system, must get
  around one obstacle: the probability of the data is intractable. The route it takes fixes its objective, its
  sampler and the evaluations it admits. Move the knobs below and watch the consequences, then open the notebooks
  to see each equation derived and checked in code.</p>
  <div class="cta"><a class="btn" href="#explorer">Open the explorer</a><a class="btn ghost" href="#chapters">Read the notebooks</a></div>
  <div class="stats"><div><b>7</b>interactive demos</div><div><b>11</b>notebook chapters</div><div><b>205</b>paper equations covered</div><div><b>402</b>checks the notebooks run</div></div>
</div></section>

<section id="chain"><div class="wrap">
  <h2>The ROSE chain</h2>
  <p class="sub">The survey follows every model through the same five questions, in order. Each tab of the explorer and each notebook chapter sits on one of them.</p>
  <ol class="chain">
__STAGES__
  </ol>
</div></section>

<section id="explorer"><div class="wrap">
  <h2>The explorer</h2>
  <p class="sub">__EXPLORER_SUB__</p>
  <div class="app-shell">
    <div class="app-bar"><i></i><i></i><i></i><span>ROSE explorer · seven tabs, one knob each</span></div>
    <div class="gallery">
__EXPLORER__
    </div>
    <div class="run">Run it yourself: <code>pip install -r app/requirements.txt && python app/app.py</code>
      <span>· or deploy <code>app/</code> as a Hugging Face Space with <code>gradio deploy</code></span></div>
  </div>
</div></section>

<section id="chapters"><div class="wrap">
  <h2>Eleven notebook chapters</h2>
  <p class="sub">Every equation in the survey and its supplement, explained twice, in plain words for a first encounter and as a refresher with assumptions and a derivation sketch, then verified in code. Each notebook runs on Colab as is; its last cell confirms that every check passed.</p>
  <div class="grid">
__CHAPTERS__
  </div>
</div></section>

<section id="verified"><div class="wrap">
  <h2>How it is verified</h2>
  <div class="how">
    <div><h3>Every equation has a check</h3><p>The 205 entries of the paper's equation registry are covered by 111 notebook entries and roughly 400 numeric or symbolic checks. A notebook fails loudly if any check fails.</p></div>
    <div><h3>The notebooks are generated</h3><p>Each notebook is built from a Python module under <code>notebooks/src/</code>. The repository's gate rebuilds them, executes them on a CPU and fails if a committed notebook drifts.</p></div>
    <div><h3>The explorer uses the same mathematics</h3><p>The demos compute their figures live from the equations the notebooks verify; the numbers under each figure come from the same computation.</p></div>
  </div>
</div></section>
</main>

<footer><div class="wrap">Part of the open artefacts of the ROSE survey of generative models. Code under MIT, text under CC BY 4.0.
  Source of this page: <code>site/build_site.py</code> (generated; do not edit <code>index.html</code> by hand).</div></footer>

<script>
(function(){
  var root=document.documentElement, btn=document.getElementById('theme');
  try{var t=localStorage.getItem('rose-theme'); if(t) root.setAttribute('data-theme',t);}catch(e){}
  btn.addEventListener('click',function(){
    var dark = root.getAttribute('data-theme')==='dark' || (!root.getAttribute('data-theme') && matchMedia('(prefers-color-scheme: dark)').matches);
    var next = dark ? 'light' : 'dark'; root.setAttribute('data-theme',next);
    try{localStorage.setItem('rose-theme',next);}catch(e){}
  });
})();
</script>
</body>
</html>
"""


def build() -> str:
    live = bool(links.SPACE_URL)
    sub = ("The app, running live. Move a slider: the figure and the numbers under it update. Each tab names the "
           "notebook chapter that derives and verifies what it shows." if live else
           "Seven interactive demos, one idea each, with the knobs that matter. Below is what each tab shows at its "
           "default settings, rendered from the app's own code; run the app to move the knobs.")
    return (TEMPLATE.replace("__SLUG__", links.REPO_SLUG).replace("__EXPLORER_SUB__", sub)
            .replace("__STAGES__", stage_cards()).replace("__CHAPTERS__", chapter_cards())
            .replace("__EXPLORER__", explorer_block())
            .replace('<div class="gallery">', '<div class="gallery">' if not live else '<div class="live">')
            .replace('A Gradio app running entirely', 'A Gradio app running entirely'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = build()
    if a.check:
        ok = OUT.exists() and OUT.read_text(encoding="utf-8") == text
        print("site/index.html current" if ok else "site/index.html STALE (run python site/build_site.py)")
        sys.exit(0 if ok else 1)
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(text) // 1024} KB)")


if __name__ == "__main__":
    main()
