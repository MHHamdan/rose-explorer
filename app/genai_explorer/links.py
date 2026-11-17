"""Where the notebooks live. One place to change when the repository is published."""
from __future__ import annotations

REPO_SLUG = "MHHamdan/rose-explorer"   # public repo holding notebooks/, app/ and site/ (the paper repo stays private until acceptance)
BRANCH = "main"
# URL of a running copy of the app (a Hugging Face Space: `cd app && gradio deploy`). Empty until deployed;
# the landing page then shows the preview gallery instead of the live app.
SPACE_URL = "https://mhamdan-rose-explorer.hf.space"

# (number, notebook file, title, one-line summary) -- tests/test_explorer.py checks these against notebooks/coverage.json
CHAPTERS = [
    (1, "01_foundations_divergences.ipynb", "Probability, entropy and divergences",
     "KL, Jensen-Shannon and Wasserstein; why forward KL covers modes and reverse KL seeks them."),
    (2, "02_likelihood_latents_bayes.ipynb", "Maximum likelihood, latent variables and Bayes",
     "Maximum likelihood is forward-KL minimisation; latent variables, Bayes' rule and EM."),
    (3, "03_variational_inference_vaes.ipynb", "Variational inference and VAEs",
     "The evidence lower bound, the reparameterisation trick, IWAE, beta-VAE and VQ-VAE."),
    (4, "04_autoregressive_models.ipynb", "Autoregressive models",
     "The chain rule, teacher forcing, KV caches, speculative decoding and perplexity."),
    (5, "05_normalizing_flows.ipynb", "Normalizing flows",
     "Exact likelihood by invertible maps: change of variables, coupling layers, dequantisation."),
    (6, "06_generative_adversarial_networks.ipynb", "Generative adversarial networks",
     "The minimax game, Jensen-Shannon at the optimum, Wasserstein critics and penalties."),
    (7, "07_energy_based_models_and_score_matching.ipynb", "Energy-based models and score matching",
     "Unnormalised densities, score matching, denoising score matching and Langevin dynamics."),
    (8, "08_continuous_time_diffusion.ipynb", "Continuous-time diffusion",
     "The forward SDE, its reversal, the probability-flow ODE and likelihood weighting."),
    (9, "09_discrete_diffusion_ddim_guidance_flow_matching.ipynb", "DDPM, DDIM, guidance and flow matching",
     "The variational bound, noise prediction, the DDIM family, classifier-free guidance, flow matching."),
    (10, "10_evaluation_metrics.ipynb", "Evaluation metrics",
     "Bits per dimension, FID and its bias, KID, precision and recall, CLIPScore, pairwise rankings."),
    (11, "11_systems_post_training_agents.ipynb", "Post-training, retrieval and agents",
     "SFT, Bradley-Terry, the KL-regularised tilt, DPO, pass@k, RAG, agents, model collapse."),
]


def colab(filename: str) -> str:
    return f"https://colab.research.google.com/github/{REPO_SLUG}/blob/{BRANCH}/notebooks/{filename}"


def github(filename: str) -> str:
    return f"https://github.com/{REPO_SLUG}/blob/{BRANCH}/notebooks/{filename}"


def chapter(n: int):
    return next(c for c in CHAPTERS if c[0] == n)
