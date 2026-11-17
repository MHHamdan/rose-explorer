"""The Gradio interface of the explorer. Shared by app/app.py (a server, e.g. a Hugging Face Space) and by the
browser-only build in site/index.html (Gradio-Lite on Pyodide); works with Gradio 5 and 6."""
from __future__ import annotations

import gradio as gr

from . import demos
from .links import CHAPTERS, chapter, colab, github

LATEX = [{"left": "$$", "right": "$$", "display": True}, {"left": "$", "right": "$", "display": False}]


def md(text: str = "", **kw):
    """gr.Markdown with inline $...$ math (Gradio renders only $$...$$ by default)."""
    return gr.Markdown(text, latex_delimiters=LATEX, **kw)


THEME = gr.themes.Soft(primary_hue="blue", secondary_hue="orange", neutral_hue="slate")
CSS = """
.rose-hero {padding: 4px 0 0 0}
.rose-chain {display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:6px 0 2px 0}
.rose-chain span.step {padding:4px 10px; border-radius:14px; background:var(--color-accent-soft); font-weight:600}
.rose-chain span.arrow {opacity:.6}
.rose-note {font-size:0.92em; opacity:.85}
footer {display:none !important}
"""

INTRO = """
<div class="rose-hero">

## ROSE explorer: how generative AI works, one knob at a time

Every generative model, from a VAE to a large language model to a text-to-image system, has to get around one
problem: the probability of the data, $p_\\theta(\\mathbf x)$, is intractable. The **route** a model takes around it
(compute it exactly, bound it, or avoid it) fixes its training **objective**, its **sampler**, and the
**evaluations** it admits. Each tab below lets you move one of those knobs and watch the consequence.

<div class="rose-chain"><span class="step">Route</span><span class="arrow">→</span><span class="step">Objective</span>
<span class="arrow">→</span><span class="step">Sampler</span><span class="arrow">→</span><span class="step">Evaluation</span>
<span class="arrow">→</span><span class="step">Systems</span></div>

</div>
"""


def _chapter_link(n: int) -> str:
    num, fname, title, _ = chapter(n)
    return (f"<span class='rose-note'>Go deeper: notebook chapter {num}, *{title}*: "
            f"[open in Colab]({colab(fname)}) · [view on GitHub]({github(fname)})</span>")


def _start_tab():
    rows = "\n".join(f"| {n} | [{t}]({colab(f)}) | {s} |" for n, f, t, s in CHAPTERS)
    md(f"""
### How to use this explorer
1. Pick a tab. Each one is a single idea from the survey, with the knobs that matter.
2. Move a slider: the figure and the numbers under it update.
3. Read the explanation, then open the matching notebook chapter to see the equation derived and **verified in
   code** (every notebook ends with a cell that checks all of its claims).

| Tab | Stage of the chain | What you will see |
|---|---|---|
| Divergences | route, objective | why likelihood training covers every mode and adversarial training may not |
| Language-model decoding | sampler | the same model, greedy vs temperature vs top-p |
| GAN critics | objective | why the original GAN loses its gradient and the Wasserstein critic does not |
| Score and Langevin | sampler | sampling from a density you cannot normalise |
| Diffusion sampler | sampler | forward noising, the reverse sampler, steps, stochasticity and guidance |
| Post-training tilt | systems | RLHF/DPO as a re-weighting of the base model, and reward over-optimisation |
| Evaluation metrics | evaluation | what FID, precision and recall can and cannot see |

### The eleven notebook chapters
Every equation in the survey, explained twice (in plain words and as a refresher) and checked in code.

| # | Chapter | Covers |
|---|---|---|
{rows}
""")


def _wire(controls, fn, plot, text, demo):
    for c in controls:
        c.change(fn, controls, [plot, text])
    demo.load(fn, controls, [plot, text])


def build() -> gr.Blocks:
    kwargs = {"title": "ROSE explorer: generative AI, one knob at a time"}
    if int(gr.__version__.split(".")[0]) < 6:      # Gradio 6 takes theme/css in launch() instead
        kwargs.update(theme=THEME, css=CSS)
    with gr.Blocks(**kwargs) as demo:
        md(INTRO)
        with gr.Tabs():
            with gr.Tab("Start here"):
                _start_tab()

            with gr.Tab("Divergences"):
                md("**Route and objective.** A model fitted by maximum likelihood minimises the *forward* KL "
                            "divergence to the data; adversarial and reward-driven training lean towards the *reverse* "
                            "one. Fit the same simple model both ways to a two-mode target.")
                with gr.Row():
                    with gr.Column(scale=1):
                        sep = gr.Slider(0.5, 8, 4, step=0.1, label="distance between the two modes")
                        w = gr.Slider(0.1, 0.9, 0.5, step=0.05, label="weight of the left mode")
                        md(_chapter_link(1))
                    with gr.Column(scale=2):
                        p1 = gr.Plot(show_label=False); t1 = md()
                _wire([sep, w], demos.divergences, p1, t1, demo)

            with gr.Tab("Language-model decoding"):
                md("**Sampler.** A language model gives a probability for every next word; a *decoding rule* "
                            "decides what to emit. A tiny bigram model trained on a few sentences shows the effect.")
                with gr.Row():
                    with gr.Column(scale=1):
                        temp = gr.Slider(0, 2, 1, step=0.05, label="temperature (0 = greedy)")
                        topp = gr.Slider(0.1, 1, 1, step=0.05, label="top-p (nucleus)")
                        topk = gr.Slider(0, 10, 0, step=1, label="top-k (0 = off)")
                        sd = gr.Slider(0, 20, 0, step=1, label="random seed")
                        md(_chapter_link(4))
                    with gr.Column(scale=2):
                        p2 = gr.Plot(show_label=False); t2 = md()
                _wire([temp, topp, topk, sd], demos.decoding, p2, t2, demo)

            with gr.Tab("GAN critics"):
                md("**Objective.** A GAN never computes $p_\\theta(\\mathbf x)$; its discriminator estimates a "
                            "divergence instead. Which divergence matters when data and samples do not overlap.")
                with gr.Row():
                    with gr.Column(scale=1):
                        sh = gr.Slider(0, 7, 1.5, step=0.05, label="how far the generator is from the data")
                        md(_chapter_link(6))
                    with gr.Column(scale=2):
                        p3 = gr.Plot(show_label=False); t3 = md()
                _wire([sh], demos.gan_divergences, p3, t3, demo)

            with gr.Tab("Score and Langevin"):
                md("**Sampler.** Energy-based and score-based models know the slope of $\\log p$ but not its "
                            "normaliser. Langevin dynamics samples with the slope alone.")
                with gr.Row():
                    with gr.Column(scale=1):
                        st = gr.Slider(0.005, 0.2, 0.05, step=0.005, label="step size")
                        ns = gr.Slider(10, 1000, 200, step=10, label="number of steps")
                        bar = gr.Slider(0.25, 5, 2, step=0.25, label="height of the barrier between the wells")
                        sd4 = gr.Slider(0, 20, 0, step=1, label="random seed")
                        md(_chapter_link(7))
                    with gr.Column(scale=2):
                        p4 = gr.Plot(show_label=False); t4 = md()
                _wire([st, ns, bar, sd4], demos.langevin, p4, t4, demo)

            with gr.Tab("Diffusion sampler"):
                md("**Sampler.** A diffusion model learns to undo noise. The noise predictor here is exact, "
                            "so what you change is only the *sampler*: the number of steps, how much fresh noise each "
                            "step injects ($\\eta$), and classifier-free guidance.")
                with gr.Row():
                    with gr.Column(scale=1):
                        tv = gr.Slider(0, 999, 300, step=1, label="forward process: time step t to display")
                        nfe = gr.Slider(2, 100, 20, step=1, label="sampler: number of steps (network evaluations)")
                        eta = gr.Slider(0, 1, 0, step=0.1, label="sampler: η (0 = deterministic ODE, 1 = stochastic)")
                        gw = gr.Slider(0, 5, 0, step=0.25, label="guidance w towards the upper-half class")
                        sd5 = gr.Slider(0, 20, 0, step=1, label="random seed")
                        md(_chapter_link(9))
                    with gr.Column(scale=2):
                        p5 = gr.Plot(show_label=False); t5 = md()
                _wire([tv, nfe, eta, gw, sd5], demos.diffusion, p5, t5, demo)

            with gr.Tab("Post-training tilt"):
                md("**Systems.** RLHF, DPO and GRPO all target the same optimum: the base model re-weighted by "
                            "$e^{r/\\beta}$. Twelve possible answers, three of them correct.")
                with gr.Row():
                    with gr.Column(scale=1):
                        beta = gr.Slider(0.05, 3, 0.5, step=0.05, label="β, the price of moving away from the base model")
                        err = gr.Checkbox(False, label="the reward model is wrong about one answer")
                        kk = gr.Slider(1, 64, 16, step=1, label="k for pass@k")
                        md(_chapter_link(11))
                    with gr.Column(scale=2):
                        p6 = gr.Plot(show_label=False); t6 = md()
                _wire([beta, err, kk], demos.tilt, p6, t6, demo)

            with gr.Tab("Evaluation metrics"):
                md("**Evaluation.** FID is the most reported number in image generation. Compare four "
                            "generators against the same real data and see what it, precision and recall each report.")
                with gr.Row():
                    with gr.Column(scale=1):
                        model = gr.Radio(["perfect copy of the data", "drops two of four modes", "blurry (too much spread)",
                                          "memorises the training set"], value="perfect copy of the data", label="generator")
                        nn = gr.Slider(50, 1600, 500, step=50, label="number of generated samples N")
                        sd7 = gr.Slider(0, 20, 0, step=1, label="random seed")
                        md(_chapter_link(10))
                    with gr.Column(scale=2):
                        p7 = gr.Plot(show_label=False); t7 = md()
                _wire([nn, model, sd7], lambda n, m, s: demos.metrics(n, m, s), p7, t7, demo)

        md("<span class='rose-note'>Part of the ROSE survey's open artefacts: every figure here is computed live "
                    "from the equations verified in the notebooks. Source: `app/` in the repository.</span>")
    return demo


def launch_kwargs() -> dict:
    return {"theme": THEME, "css": CSS} if int(gr.__version__.split(".")[0]) >= 6 else {}
