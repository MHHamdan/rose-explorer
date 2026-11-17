"""Learner aids appended to every chapter by nbkit.Chapter.build(): key takeaways and self-check questions.

Keyed by chapter number. Each question has a short answer shown in a collapsed <details> block, so a reader can
test themselves before looking. Keep answers to what the chapter's entries establish.
"""

LEARNING = {
    1: {
        "takeaways": [
            "The KL divergence $D_{\\mathrm{KL}}(p\\,\\Vert\\,q)$ is never negative and is zero only when $p=q$; it is not symmetric.",
            "Forward KL $D_{\\mathrm{KL}}(p_{\\mathrm{data}}\\Vert q)$ punishes $q$ for missing data, so it covers every mode; reverse KL punishes $q$ for mass where there are no data, so it picks modes.",
            "Jensen--Shannon is symmetric and bounded by $\\log 2$, but it saturates when two distributions do not overlap.",
            "Wasserstein-1 measures how far mass must move, so it keeps decreasing as two non-overlapping distributions approach each other.",
        ],
        "questions": [
            ("Why can $D_{\\mathrm{KL}}(p_{\\mathrm{data}}\\Vert q)$ be infinite?",
             "If $q$ gives zero probability to a region where the data have positive probability, the log-ratio is infinite there."),
            ("Two distributions sit on disjoint intervals. You slide one towards the other without overlap. What happens to JS and to $W_1$?",
             "JS stays at $\\log 2$ throughout; $W_1$ decreases with the distance between them."),
            ("A single Gaussian is fitted to a two-mode target. Which divergence puts the Gaussian between the modes?",
             "The forward KL (maximum likelihood): it must cover both modes, so it centres between them with a wide variance."),
        ],
    },
    2: {
        "takeaways": [
            "Maximum likelihood on a large sample maximises $\\mathbb E_{p_{\\mathrm{data}}}[\\log p_\\theta]$, which is the same as minimising $D_{\\mathrm{KL}}(p_{\\mathrm{data}}\\Vert p_\\theta)$.",
            "So every likelihood-trained model inherits the mode-covering behaviour of the forward KL.",
            "A latent-variable model's likelihood is an integral over the latent, which is usually intractable; Bayes' rule gives the posterior only up to that same integral.",
            "EM never decreases the likelihood: each step maximises a lower bound that touches the likelihood at the current parameters.",
        ],
        "questions": [
            ("Why does the entropy of the data not matter when we maximise the likelihood?",
             "$D_{\\mathrm{KL}}(p_{\\mathrm{data}}\\Vert p_\\theta)=-H(p_{\\mathrm{data}})-\\mathbb E_{p_{\\mathrm{data}}}[\\log p_\\theta]$, and $H(p_{\\mathrm{data}})$ does not depend on $\\theta$."),
            ("What makes $p_\\theta(\\mathbf x)=\\int p_\\theta(\\mathbf x\\mid\\mathbf z)p(\\mathbf z)\\,d\\mathbf z$ hard to compute?",
             "The integral over a high-dimensional latent has no closed form for a neural-network decoder."),
            ("In EM, why can the log-likelihood not go down after an iteration?",
             "The E-step makes the lower bound equal to the log-likelihood; the M-step raises the bound; the log-likelihood is at least the bound."),
        ],
    },
    3: {
        "takeaways": [
            "The ELBO is a lower bound on $\\log p_\\theta(\\mathbf x)$; the gap is exactly the KL from the encoder to the true posterior.",
            "The reparameterisation trick writes $\\mathbf z=\\mu+\\sigma\\odot\\epsilon$ so that gradients flow through the sampling step.",
            "Importance-weighted bounds (IWAE) tighten monotonically as the number of samples $K$ grows.",
            "Weighting the KL term by $\\beta>1$ gives a looser bound; $\\beta$ is best read as a control on how much information the latent carries.",
        ],
        "questions": [
            ("When is the ELBO equal to $\\log p_\\theta(\\mathbf x)$?",
             "When the encoder $q_\\phi(\\mathbf z\\mid\\mathbf x)$ equals the true posterior $p_\\theta(\\mathbf z\\mid\\mathbf x)$."),
            ("Why do VAE samples tend to be blurry?",
             "A factorised Gaussian decoder must explain many inputs that map to the same latent with one mean, and the maximum-likelihood answer is their average."),
            ("Is a VAE's reported ELBO comparable to a flow's log-likelihood?",
             "Only as a bound: the ELBO is lower than the VAE's true log-likelihood by an unknown gap, so a difference mixes model quality with bound tightness."),
        ],
    },
    4: {
        "takeaways": [
            "The chain rule writes any sequence probability as a product of next-element conditionals, so autoregressive models compute the likelihood exactly.",
            "Training uses the true prefix at every position (teacher forcing) in one parallel pass; sampling must go one element at a time.",
            "A key--value cache avoids recomputing past keys and values, at a memory cost that grows linearly with context length.",
            "Perplexity depends on the tokeniser; bits per byte removes it, so compare models across tokenisers in bits per byte.",
        ],
        "questions": [
            ("Why is training an autoregressive transformer much cheaper than sampling from it?",
             "Training scores all positions in one masked pass; sampling needs one network evaluation per generated element."),
            ("Speculative decoding uses a small draft model. Does it change what the large model would have generated?",
             "No: the accept-or-resample rule leaves the output distribution exactly equal to the large model's."),
            ("Model A has perplexity 12 and model B 15, with different tokenisers. Which is better?",
             "You cannot tell from perplexity; convert both to bits per byte of the same text."),
        ],
    },
    5: {
        "takeaways": [
            "A flow is an invertible map; the change-of-variables formula gives the exact density using the log-determinant of its Jacobian.",
            "Coupling and autoregressive layers have triangular Jacobians, so the determinant costs $O(D)$ instead of $O(D^3)$.",
            "On 8-bit images a continuous density gives only a bound on the discrete code length, through dequantisation.",
            "Data on a lower-dimensional surface push the log-determinant to $-\\infty$, which makes flow likelihoods ill-conditioned.",
        ],
        "questions": [
            ("Why must a flow layer be invertible?",
             "The change-of-variables formula maps each data point to exactly one latent point; without invertibility the density is not defined this way."),
            ("What makes a coupling layer cheap?",
             "Half the coordinates pass unchanged, so the Jacobian is triangular and its determinant is the product of the diagonal."),
            ("A flow reports a better likelihood on another dataset than on its training data. Is that a contradiction?",
             "No: likelihood measures local density, not resemblance, and flows are known to assign high likelihood to some out-of-distribution inputs."),
        ],
    },
    6: {
        "takeaways": [
            "A GAN never evaluates a density; it trains a generator against a discriminator that estimates a divergence.",
            "With the optimal discriminator, the original game minimises $2\\,\\mathrm{JS}-\\log 4$, which saturates when data and samples do not overlap.",
            "The non-saturating loss keeps gradients alive early in training; at the optimal discriminator it pushes towards a reverse KL, which favours dropping modes.",
            "Wasserstein critics need a Lipschitz constraint, enforced by gradient penalties or spectral normalisation; $R_1$ on real data helps convergence.",
        ],
        "questions": [
            ("What does the optimal discriminator output where real and generated data are equally likely?",
             "$D^*=\\tfrac12$."),
            ("Why does the original GAN generator stop learning when its samples are far from the data?",
             "With non-overlapping supports the discriminator is perfect, JS is stuck at $\\log 2$, and the generator's gradient vanishes."),
            ("Why do GANs tend to drop modes?",
             "The non-saturating objective behaves like a reverse KL, which is not penalised for ignoring parts of the data."),
        ],
    },
    7: {
        "takeaways": [
            "An energy-based model defines $p_\\theta\\propto e^{-E_\\theta}$; the normaliser $Z_\\theta$ is intractable, but the score $-\\nabla E$ does not need it.",
            "The likelihood gradient is a data term minus a model term; the model term needs samples from the model.",
            "Score matching fits the score directly; denoising score matching does so by predicting the noise added to data.",
            "Langevin dynamics samples with only the score, but crosses high-energy barriers slowly; annealing through noise levels fixes this and leads to diffusion.",
        ],
        "questions": [
            ("Why does the score $\\nabla_{\\mathbf x}\\log p$ not depend on the normaliser?",
             "$\\log Z$ is a constant in $\\mathbf x$, so its gradient is zero."),
            ("What does a denoising-score-matching network learn: the score of the data or of the noisy data?",
             "The score of the data smoothed by the noise; it approaches the data score only as the noise level goes to zero."),
            ("Langevin chains all started in one well rarely reach the other. Which knob helps?",
             "Lowering the barrier (or annealing from high noise), or running many more steps; a larger step size also helps but adds discretisation error."),
        ],
    },
    8: {
        "takeaways": [
            "A forward SDE turns data into noise; its marginals obey the Fokker--Planck equation.",
            "The time reversal is another SDE whose only unknown is the score of the noisy marginals, with coefficient $g^2$.",
            "The probability-flow ODE shares the same marginals with no noise, making the model a continuous flow with an exact likelihood.",
            "With the weighting $\\lambda=g^2$ the training loss bounds the negative log-likelihood of the reverse-SDE model.",
        ],
        "questions": [
            ("What is the one quantity a network must learn to run the reverse SDE?",
             "The score $\\nabla_{\\mathbf x}\\log p_t(\\mathbf x)$ at every time $t$."),
            ("Do the reverse SDE and the probability-flow ODE generate the same distribution?",
             "With the exact score, yes; with a learned score they define different models."),
            ("Why does the paper warn against mixing variational bounds and ODE likelihoods in one table?",
             "They measure different models (stochastic vs deterministic sampler), so their difference is not a model-quality difference."),
        ],
    },
    9: {
        "takeaways": [
            "DDPM's forward process has a closed-form marginal $q(\\mathbf x_t\\mid\\mathbf x_0)$, so any noise level can be sampled in one step.",
            "Predicting the noise $\\epsilon$ is equivalent to estimating the score, and the simple loss is a reweighted variational bound, not a bound itself.",
            "DDIM samplers with any $\\eta$ keep the training marginals; $\\eta=0$ is deterministic, $\\eta=1$ recovers ancestral sampling.",
            "Classifier-free guidance with $w>1$ sharpens the conditional at the cost of diversity; flow matching regresses a velocity along a straight path.",
        ],
        "questions": [
            ("Why can DDPM training pick a random time step for each example?",
             "Because $q(\\mathbf x_t\\mid\\mathbf x_0)$ is Gaussian in closed form, so $\\mathbf x_t$ is sampled directly from $\\mathbf x_0$."),
            ("Does training with $\\mathcal L_{\\mathrm{simple}}$ give a likelihood bound?",
             "No: it reweights the bound's terms; bits per dimension must be computed from the variational bound itself."),
            ("Which guidance weight gives the plain conditional model?",
             "$w=1$; $w=0$ is unconditional and $w>1$ exaggerates the condition."),
        ],
    },
    10: {
        "takeaways": [
            "Bits per dimension on 8-bit images is an upper bound on the code length; differences mix model quality with bound tightness.",
            "FID is biased upward roughly as $c/N$: values computed with different sample counts are not comparable.",
            "Precision and recall separate realism from coverage; FID mixes them, and none of these metrics detects copying of training data.",
            "CLIPScore measures agreement with the prompt, not image quality; pairwise human or model judgements need position and length controls.",
        ],
        "questions": [
            ("Model A has FID 10 with 10k samples, model B FID 9 with 50k samples. Is B better?",
             "Not established: B's FID is biased less because it used more samples; recompute both at the same $N$."),
            ("A generator drops half the classes but renders the rest perfectly. What happens to precision and recall?",
             "Precision stays high, recall falls."),
            ("Can a model that outputs copies of its training images get a good FID?",
             "Yes: FID, precision and recall cannot tell copies from new samples; a nearest-neighbour memorisation check is needed."),
        ],
    },
    11: {
        "takeaways": [
            "Pretraining and supervised fine-tuning are both maximum likelihood with the chain rule; they differ in data and conditioning.",
            "The KL-regularised reward objective has an exact optimum: the reference model re-weighted by $e^{r/\\beta}$, which keeps its support.",
            "With a correct 0/1 verifier this exact tilt cannot lower pass@k; trained policies can, because they are not the exact tilt.",
            "DPO inverts the tilt to train without a reward model; retrieval is a truncated latent-variable marginal; an agent's trajectory is the chain rule with an external environment.",
        ],
        "questions": [
            ("Can RLHF's optimal policy produce an answer the reference model never produces?",
             "No: the tilt multiplies the reference probability, so zero stays zero."),
            ("What happens to the tilted policy as $\\beta\\to0$ when the reward model is wrong about one answer?",
             "All mass moves to the answer with the highest reward, including the wrong one: reward over-optimisation."),
            ("Why does refitting a model on its own samples, generation after generation, narrow its distribution?",
             "Each refit underestimates the variance on average by a factor $1-1/n$, and the errors compound."),
        ],
    },
}
