---
title: ROSE explorer
emoji: 🌹
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
license: mit
---

# ROSE explorer

Interactive demos of the mathematics of the ROSE survey of generative models, one knob at a time: divergences,
language-model decoding, GAN critics, Langevin sampling, diffusion samplers, post-training tilts and evaluation
metrics. Each tab names the notebook chapter (`../notebooks/`) that explains and verifies its equations.

```bash
python app/app.py                 # run locally (Gradio 5 or 6)
cd app && gradio deploy           # publish as a Hugging Face Space, then set SPACE_URL in genai_explorer/links.py
```

The folder is self-contained (`app.py`, `genai_explorer/`, `requirements.txt`): deploy it as is.
