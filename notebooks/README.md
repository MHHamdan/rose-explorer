# The mathematics of the survey, one chapter per topic

These notebooks explain and verify every equation of the paper and its supplement: 205 of the
205 entries in [`data/equation_registry.json`](../data/equation_registry.json). Each equation is given
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
| 1 | [Probability, entropy and divergences](01_foundations_divergences.ipynb) | 9 | 14 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/01_foundations_divergences.ipynb) |
| 2 | [Maximum likelihood, latent variables and Bayes](02_likelihood_latents_bayes.ipynb) | 7 | 9 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/02_likelihood_latents_bayes.ipynb) |
| 3 | [Variational inference and variational autoencoders](03_variational_inference_vaes.ipynb) | 8 | 17 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/03_variational_inference_vaes.ipynb) |
| 4 | [Autoregressive models: the chain rule, teacher forcing and perplexity](04_autoregressive_models.ipynb) | 7 | 15 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/04_autoregressive_models.ipynb) |
| 5 | [Normalizing flows: exact likelihood by invertible transport](05_normalizing_flows.ipynb) | 12 | 20 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/05_normalizing_flows.ipynb) |
| 6 | [Generative adversarial networks: implicit models and their divergences](06_generative_adversarial_networks.ipynb) | 8 | 18 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/06_generative_adversarial_networks.ipynb) |
| 7 | [Energy-based models, score matching and Langevin sampling](07_energy_based_models_and_score_matching.ipynb) | 12 | 23 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/07_energy_based_models_and_score_matching.ipynb) |
| 8 | [Continuous-time diffusion: SDEs, their reversal and the probability-flow ODE](08_continuous_time_diffusion.ipynb) | 6 | 12 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/08_continuous_time_diffusion.ipynb) |
| 9 | [Discrete-time diffusion: DDPM, DDIM, guidance and flow matching](09_discrete_diffusion_ddim_guidance_flow_matching.ipynb) | 14 | 31 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/09_discrete_diffusion_ddim_guidance_flow_matching.ipynb) |
| 10 | [Evaluation: what each metric measures](10_evaluation_metrics.ipynb) | 16 | 22 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/10_evaluation_metrics.ipynb) |
| 11 | [From models to systems: post-training, retrieval and agents](11_systems_post_training_agents.ipynb) | 12 | 24 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MHHamdan/rose-explorer/blob/main/notebooks/11_systems_post_training_agents.ipynb) |

All registry entries are covered.

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

The Colab links use the repository slug `MHHamdan/rose-explorer`, filled in when the repository is published.
