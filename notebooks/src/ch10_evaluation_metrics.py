"""Chapter 10: Evaluation -- what each metric measures. Registry ids: notebooks/src/chapter_ids.json."""
from nbkit import Chapter


def build():
    c = Chapter(
        10, "evaluation_metrics", "Evaluation: what each metric measures",
        summary=(
            "A generative model is judged by numbers: bits per dimension, perplexity, the Inception Score, FID, KID, "
            "precision and recall, CLIPScore, win rates. Each of these numbers is an estimator of some quantity under "
            "some assumptions, and the paper's comparison contract (Section 7) says which values may be compared. "
            "This chapter states each metric as an equation, explains what it estimates and what it cannot see, and "
            "verifies the identities and the biases on small examples where the truth is known."),
        prerequisites="Chapter 1 (KL divergence, Jensen's inequality) and Chapter 3 (the ELBO and IWAE)",
        paper_sections="main §7; Supplement S1.12 and S5")

    # ------------------------------------------------------------------------------------------ likelihood
    c.section("Likelihood-based quantities",
              "Models that compute or bound $\\log p_\\theta(\\mathbf{x})$ report it in a unit that is comparable across "
              "datasets of different size: bits per dimension for images, perplexity or bits per byte for text. Both "
              "hide a convention that must be stated before two values are compared.")

    c.eq(["EQ-S5-001"], "The uniform dequantisation bound",
         r"\mathbf{u}\sim\mathcal{U}[0,1)^D,\qquad \mathbb{E}_{\mathbf{u}}\big[\log p_\theta(\mathbf{x}+\mathbf{u})\big]\;\le\;\log P_\theta(\mathbf{x}),"
         r"\qquad P_\theta(\mathbf{x})=\int_{[0,1)^D} p_\theta(\mathbf{x}+\mathbf{u})\,d\mathbf{u}",
         plain=(
             "Image pixels are whole numbers from 0 to 255, but most models (flows, diffusion models, VAEs with a "
             "Gaussian decoder) define a density over real numbers. To make the two meet, one adds a random fraction "
             "$\\mathbf{u}$ between 0 and 1 to every pixel before scoring it. The equation says that the average "
             "log-density one then gets is never larger than the log of the probability the model assigns to the whole "
             "unit cube around the integer image: the continuous number is a pessimistic estimate of the discrete one."),
         refresher=(
             "$P_\\theta(\\mathbf{x})$ is the mass the continuous density puts on the unit cube at the integer point "
             "$\\mathbf{x}$, which is the probability of the discrete image under the model's induced discrete "
             "distribution. Since $\\log$ is concave, Jensen gives $\\mathbb{E}_{\\mathbf{u}}[\\log p_\\theta(\\mathbf{x}+\\mathbf{u})]\\le"
             "\\log\\mathbb{E}_{\\mathbf{u}}[p_\\theta(\\mathbf{x}+\\mathbf{u})]=\\log P_\\theta(\\mathbf{x})$ (Theis et al. 2016; "
             "Dinh et al. 2017). Equality holds iff $p_\\theta$ is constant on the cube, as for a histogram density. "
             "Variational dequantisation (Ho et al. 2019) replaces the uniform $\\mathbf{u}$ by a learned "
             "$q(\\mathbf{u}\\mid\\mathbf{x})$ and tightens the bound, so values under the two conventions are not "
             "strictly comparable. The check below computes both sides exactly for a one-pixel model: a histogram "
             "density (tight) and a smooth Beta density (strict)."),
         code=r'''
# One pixel with 256 levels. Model A: a histogram density, constant on each cell. Model B: a smooth Beta density.
P = rng.dirichlet(np.ones(256))                       # discrete probabilities of model A
dens_A = lambda y: 256 * P[np.minimum((y * 256).astype(int), 255)]   # density on y in [0,1)
x = 77
u = (np.arange(20_000) + 0.5) / 20_000                 # a fine deterministic grid over the unit interval
lhs_A = np.mean(np.log(dens_A((x + u) / 256) / 256))   # E_u log p(x+u) in the pixel-scale density
check("histogram density: Jensen is tight", *close(lhs_A, np.log(P[x]), 1e-9, "E_u log p(x+u) vs log P(x) "))

a_, b_ = 2.0, 5.0
P_B = scipy.stats.beta.cdf((x + 1) / 256, a_, b_) - scipy.stats.beta.cdf(x / 256, a_, b_)   # exact cube mass
lhs_B = np.mean(scipy.stats.beta.logpdf((x + u) / 256, a_, b_) - np.log(256))
check("smooth density: E_u log p(x+u) < log P(x)", lhs_B < np.log(P_B),
      f"{lhs_B:.5f} < {np.log(P_B):.5f}, gap {np.log(P_B) - lhs_B:.2e} nats")
''')

    c.eq(["EQ-M-044", "EQ-S1-057"], "Bits per dimension",
         r"\mathrm{bits/dim}(\mathbf{x})=\frac{-\,\mathbb{E}_{\mathbf{u}}[\log p_\theta(\mathbf{y})]+D\log 256}{D\log 2},"
         r"\qquad \mathbf{y}=\frac{\mathbf{x}+\mathbf{u}}{256}",
         plain=(
             "Take the model's average log-density of the dequantised image, flip its sign (so that better is "
             "smaller), add the $D\\log 256$ that comes from rescaling the pixels from $[0,256)$ to $[0,1)$, and divide "
             "by the number of pixels $D$ and by $\\log 2$ to turn nats into bits. The result is the number of bits "
             "per pixel the model would need to encode the image, and it can be compared across image sizes."),
         refresher=(
             "The change of variables $\\mathbf{y}=(\\mathbf{x}+\\mathbf{u})/256$ multiplies the density by $256^D$, "
             "hence the $D\\log 256$ term. By the previous entry, $-\\mathbb{E}_{\\mathbf{u}}\\log p_\\theta(\\mathbf{y})+D\\log256"
             "\\ge-\\log P_\\theta(\\mathbf{x})$, so bits/dim is an upper bound on the code length per pixel of the "
             "model's own discrete distribution, with equality for a histogram density. A VAE replaces $\\log p_\\theta$ "
             "by its ELBO or IWAE bound and a diffusion model by its variational bound, which adds a second layer of "
             "slack. The check reproduces `bits_per_dim_dequantisation_and_unit_conversions` from the repository: "
             "for the histogram density, bits/dim equals $-\\log_2 P(x)$ to machine precision; for the smooth density "
             "it is strictly larger."),
         code=r'''
bpd_A = (-np.mean(np.log(dens_A((x + u) / 256))) + np.log(256)) / np.log(2)
check("bits/dim of a histogram density = -log2 P(x)", *close(bpd_A, -np.log2(P[x]), 1e-9))
bpd_B = (-np.mean(scipy.stats.beta.logpdf((x + u) / 256, a_, b_)) + np.log(256)) / np.log(2)
check("bits/dim of a smooth density >= -log2 P(x)", bpd_B >= -np.log2(P_B) - 1e-9,
      f"{bpd_B:.4f} bits >= {-np.log2(P_B):.4f} bits")
''')

    c.eq(["EQ-M-045", "EQ-S1-058", "EQ-S5-002"], "Every reported bits/dim bounds the same discrete code length",
         r"\mathrm{bpd}_{\mathrm{AR}}=-\tfrac{1}{D}\log_2 P_\theta(\mathbf{x})\ \text{(exact)},\qquad"
         r"\mathrm{bpd}_{\mathrm{flow}},\ \mathrm{bpd}_{\mathrm{VAE}},\ \mathrm{bpd}_{\mathrm{diffusion}}\ \ge\ -\tfrac{1}{D}\log_2 P_\theta(\mathbf{x})",
         plain=(
             "An autoregressive model over discrete pixels reports its exact code length. A flow reports the "
             "dequantisation bound, a VAE adds the ELBO gap on top, a diffusion model adds its own bound. Each number "
             "is an honest upper bound on *its own* model's code length, so all of them can be compared *as bounds*, "
             "but the gap between two of them mixes how good the models are with how loose each bound is."),
         refresher=(
             "Write $B_m\\ge C_m$ for model $m$'s reported bound and true code length. $B_1<B_2$ implies $C_1<B_2$ but "
             "says nothing about $C_1$ versus $C_2$. The paper's wording fix (registry status YES-WITH-WORDING-FIX) "
             "replaced 'all values are comparable' by 'comparable as bounds'. The check builds two one-pixel models "
             "where the ranking by bound and by true code length disagree: a flow-like smooth density with the "
             "tighter true code length but a looser dequantisation slack loses on the reported number. A VAE's "
             "ELBO slack is the posterior KL of Chapter 3, which only widens the picture."),
         code=r'''
# Two models of the same pixel distribution. True code lengths C (bits) and reported bounds B (bits).
# Model 1: a smooth Beta(2,5)-shaped density; model 2: a histogram model whose cell mass at x equals model 1's
# mass times 0.9 (slightly worse true code length, but no dequantisation slack).
C1 = -np.log2(P_B)
B1 = bpd_B
gap_bits = B1 - C1          # model 1's dequantisation slack, in bits
P2 = P_B * 2 ** (-gap_bits / 2)   # model 2's true code length sits halfway inside that slack
C2 = -np.log2(P2)
B2 = C2                     # histogram: the bound is tight
check("model 1 has the smaller true code length", C1 < C2, f"C1 = {C1:.4f} < C2 = {C2:.4f}")
check("but the larger reported bound", B1 > B2, f"B1 = {B1:.4f} > B2 = {B2:.4f}: the ranking by bounds reverses")
check("each bound is >= its own code length", B1 >= C1 - 1e-12 and B2 >= C2 - 1e-12)
''')

    c.eq(["EQ-S5-003"], "The importance-weighted bound tightens the ELBO",
         r"\log p_\theta(\mathbf{x})\ \ge\ \mathbb{E}_{\mathbf{z}_{1:k}\sim q_\phi}\Big[\log\frac1k\sum_{i=1}^{k}\frac{p_\theta(\mathbf{x},\mathbf{z}_i)}{q_\phi(\mathbf{z}_i\mid\mathbf{x})}\Big]=:\mathcal{L}_k,"
         r"\qquad \mathcal{L}_1\le\mathcal{L}_k\le\mathcal{L}_{k+1}\le\log p_\theta(\mathbf{x}),\quad \mathcal{L}_k\to\log p_\theta(\mathbf{x})",
         plain=(
             "A VAE cannot compute its likelihood, only a lower bound. Averaging $k$ importance weights before taking "
             "the logarithm gives a better bound, and more samples never make it worse; with enough samples it "
             "reaches the true value. The paper's benchmark reports the $k=100$ bound so that VAE numbers are closer "
             "to a likelihood than to an ELBO."),
         refresher=(
             "Burda et al. (2016). $\\mathcal{L}_k\\le\\log p$ is Jensen applied to the unbiased estimator "
             "$\\frac1k\\sum_i w_i$ of $p_\\theta(\\mathbf{x})$; monotonicity in $k$ follows because the average of $k$ "
             "weights is the expectation of the average of a random $(k-1)$-subset, and Jensen again. Convergence "
             "needs $w_i$ bounded, or at least finite variance for a useful rate. Chapter 3 verifies the family in a "
             "Gaussian model; here the check is the one that matters for evaluation: the gap to the exact likelihood "
             "shrinks with $k$ and is positive at $k=100$, so an IWAE-100 number is still a bound."),
         code=r'''
# Exact 1-D model: p(z) = N(0,1), p(x|z) = N(z, 1), so log p(x) = log N(x; 0, 2). Encoder q(z|x) = N(x/2, 1) (too wide).
xv = 1.3
logp = scipy.stats.norm.logpdf(xv, 0, np.sqrt(2))
def L_k(k, n_rep=4000):
    z = rng.normal(xv / 2, 1.0, size=(n_rep, k))
    logw = scipy.stats.norm.logpdf(z, 0, 1) + scipy.stats.norm.logpdf(xv, z, 1) - scipy.stats.norm.logpdf(z, xv / 2, 1)
    return float(np.mean(scipy.special.logsumexp(logw, axis=1) - np.log(k)))
Ls = {k: L_k(k) for k in (1, 5, 20, 100)}
check("L_1 <= L_5 <= L_20 <= L_100", Ls[1] <= Ls[5] <= Ls[20] <= Ls[100], {k: round(v, 4) for k, v in Ls.items()})
check("every L_k <= log p(x)", all(v <= logp + 2e-3 for v in Ls.values()), f"log p(x) = {logp:.4f}")
check("L_100 is still a bound (gap > 0)", logp - Ls[100] > 0, f"gap at k=100: {logp - Ls[100]:.4f} nats")
''')

    c.eq(["EQ-S5-004"], "Weighted diffusion objectives are ELBOs under noise augmentation",
         r"\mathcal{L}_w=\int w(\lambda)\,\tfrac{d}{d\lambda}\mathcal{L}_{\mathrm{ELBO}}(\theta;\lambda)\,d\lambda=\int p_w(\lambda_0)\,\mathcal{L}_{\mathrm{ELBO}}(\theta;\lambda_0)\,d\lambda_0+\text{const},\quad p_w=-\tfrac{dw}{d\lambda}\ge0\iff w\text{ non-increasing}",
         plain=(
             "Diffusion models are usually trained with a loss that weights the noise levels differently from the "
             "likelihood bound. Kingma and Gao (2023) showed that if the weighting never increases with the "
             "signal-to-noise ratio, the training loss is an average of likelihood bounds for data blurred by extra "
             "Gaussian noise. Other weightings have no such reading. Weightings that give good "
             "samples and weightings that give good likelihoods differ, which is why one network can report two "
             "different numbers."),
         refresher=(
             "Write the per-level ELBO as a function of log-SNR $\\lambda$; its $\\lambda$-derivative is the "
             "denoising loss at that level. A weighting $w(\\lambda)$ that is non-increasing in $\\lambda$ can be "
             "written as the tail integral of a density $p(\\lambda_0)$ over 'augmentation noise levels', and "
             "$\\int w\\,\\mathcal{L}'\\,d\\lambda=\\int p(\\lambda_0)\\,\\mathcal{L}_{\\mathrm{ELBO}}(\\lambda_0)\\,d\\lambda_0$ by "
             "integration by parts, an average of ELBOs of noise-augmented data. Non-monotone weightings have no "
             "such reading. The check verifies the integration-by-parts identity for a monotone weight on a toy "
             "per-level loss and shows that a non-monotone weight has no non-negative mixing density."),
         code=r'''
lam = np.linspace(-8, 8, 4001)
ell = lambda l: 0.5 * np.log1p(np.exp(l))            # a stand-in per-level ELBO term, increasing in log-SNR
dell = np.gradient(ell(lam), lam)                      # its derivative = the per-level denoising loss
w_mono = scipy.special.expit(-lam)                     # monotone non-increasing weight
p_mix = -np.gradient(w_mono, lam)                      # the implied density over augmentation levels (>= 0)
lhs = np.trapezoid(w_mono * dell, lam)
rhs = np.trapezoid(p_mix * (ell(lam) - ell(lam[0])), lam) + w_mono[-1] * (ell(lam[-1]) - ell(lam[0]))
check("monotone weight: weighted loss = mixture of ELBOs (integration by parts)", *close(lhs, rhs, 1e-6))
check("implied mixing density is non-negative", np.all(p_mix >= -1e-12), f"min p = {p_mix.min():.2e}")
w_bump = np.exp(-lam ** 2 / 2)                          # a non-monotone (sample-quality-style) weight
check("non-monotone weight has a negative 'density' somewhere", np.min(-np.gradient(w_bump, lam)) < -1e-3,
      f"min = {np.min(-np.gradient(w_bump, lam)):.3f}: no ELBO reading")
''')

    # ------------------------------------------------------------------------------------------ feature-space
    c.section("Feature-space fidelity",
              "Models without a likelihood are compared through the statistics of a pretrained network's features on "
              "generated and real samples. The estimators below each carry a bias or a blind spot that decides whether "
              "two published numbers may be compared.")

    c.eq(["EQ-S5-006"], "The Inception Score",
         r"\mathrm{IS}=\exp\Big(\mathbb{E}_{\mathbf{x}\sim p_\theta}\big[D_{\mathrm{KL}}\big(p(y\mid\mathbf{x})\,\|\,p(y)\big)\big]\Big),\qquad p(y)=\mathbb{E}_{\mathbf{x}\sim p_\theta}[p(y\mid\mathbf{x})]",
         plain=(
             "Run a classifier on each generated image. A good image gets a confident prediction (a peaked "
             "$p(y\\mid\\mathbf{x})$); a diverse set of images spreads the predictions over many classes (a flat marginal "
             "$p(y)$). The average KL between the two is large exactly when both hold, and the exponential turns it "
             "into an 'effective number of classes'. The score never looks at real images."),
         refresher=(
             "$\\mathbb{E}_{\\mathbf{x}}D_{\\mathrm{KL}}(p(y|\\mathbf{x})\\|p(y))=I(\\mathbf{x};y)=H(y)-H(y|\\mathbf{x})$, the mutual "
             "information between image and predicted class, so $1\\le\\mathrm{IS}\\le$ number of classes, with the "
             "maximum for a generator that emits one perfectly classified prototype per class, uniformly, which is "
             "the gaming Borji (2019) describes. The value depends on the classifier weights and on the number of "
             "splits; Barratt and Sharma (2018) and Chong and Forsyth (2020) document the implementation and "
             "sample-size dependence. The check verifies the mutual-information identity and the two extremes."),
         code=r'''
def inception_score(post):                             # post: (N, C) class posteriors of generated images
    py = post.mean(0)
    kl = np.sum(post * (np.log(post + 1e-300) - np.log(py + 1e-300)), axis=1)
    return float(np.exp(kl.mean()))
C_ = 10
post = rng.dirichlet(np.ones(C_) * 0.3, size=5000)
H = lambda p: -np.sum(p * np.log(p + 1e-300), axis=-1)
mi = H(post.mean(0)) - H(post).mean()
check("log IS = H(y) - H(y|x) (mutual information)", *close(np.log(inception_score(post)), mi, 1e-9))
proto = np.eye(C_)[rng.integers(0, C_, 5000)]           # one perfectly classified prototype per class, uniformly
check("one prototype per class attains IS = number of classes", *close(inception_score(proto), C_, 0.05, "IS "))
same = np.tile(rng.dirichlet(np.ones(C_)), (5000, 1))    # every image gets the same posterior
check("identical posteriors give IS = 1", *close(inception_score(same), 1.0, 1e-9))
''')

    c.eq(["EQ-M-047", "EQ-S5-007"], "The Fréchet Inception Distance",
         r"\mathrm{FID}=\|\mu_r-\mu_g\|_2^2+\mathrm{Tr}\big(\Sigma_r+\Sigma_g-2(\Sigma_r\Sigma_g)^{1/2}\big)",
         plain=(
             "Fit a Gaussian to the features of real images (mean $\\mu_r$, covariance $\\Sigma_r$) and another to the "
             "features of generated images. FID is the squared distance between the two Gaussians: how far apart the "
             "means are, plus how different the spreads are. Zero means the two feature clouds have the same mean and "
             "covariance; it does not mean the images are indistinguishable."),
         refresher=(
             "This is the squared 2-Wasserstein distance between $\\mathcal{N}(\\mu_r,\\Sigma_r)$ and "
             "$\\mathcal{N}(\\mu_g,\\Sigma_g)$ (Dowson and Landau 1982), reported by Heusel et al. (2017) on Inception "
             "pool features. For commuting covariances the trace term is $\\sum_i(\\sqrt{\\lambda_i}-\\sqrt{\\nu_i})^2$. "
             "$(\\Sigma_r\\Sigma_g)^{1/2}$ is not symmetric, but its trace is, so FID is symmetric in its arguments. "
             "The check reproduces `fid_formula_and_sample_size_bias`: the 1-D closed form, the diagonal case, and "
             "symmetry under a non-commuting pair."),
         code=r'''
def fid(m1, C1, m2, C2):
    cs = scipy.linalg.sqrtm(C1 @ C2).real
    return float(np.sum((m1 - m2) ** 2) + np.trace(C1 + C2 - 2 * cs))
check("1-D: (mu1-mu2)^2 + (s1-s2)^2", *close(fid(np.array([1.0]), np.array([[4.0]]), np.array([0.0]), np.array([[1.0]])), 2.0, 1e-12))
d1, d2 = rng.random(5) + 0.5, rng.random(5) + 0.5
check("diagonal covariances: sum (sqrt l_i - sqrt n_i)^2",
      *close(fid(np.zeros(5), np.diag(d1), np.zeros(5), np.diag(d2)), np.sum((np.sqrt(d1) - np.sqrt(d2)) ** 2), 1e-10))
A, B = rng.normal(size=(5, 5)), rng.normal(size=(5, 5))
C1, C2 = A @ A.T + np.eye(5), B @ B.T + np.eye(5)
check("symmetric although sqrtm(C1 C2) is not", *close(fid(np.zeros(5), C1, np.ones(5), C2), fid(np.ones(5), C2, np.zeros(5), C1), 1e-8))
check("zero for identical Gaussians", *close(fid(np.ones(5), C1, np.ones(5), C1), 0.0, 1e-8))
''')

    c.eq(["EQ-S5-008"], "FID is biased upward, roughly as $1/N$",
         r"\mathbb{E}\big[\widehat{\mathrm{FID}}_N\big]\ \approx\ \mathrm{FID}_\infty+\frac{c}{N}",
         plain=(
             "FID is computed from a finite sample, and the sample means and covariances carry noise. That noise "
             "always makes the two fitted Gaussians look more different than they are, so the estimate is too high, "
             "by an amount that shrinks like one over the number of samples. A 10k-sample FID and a 50k-sample FID of "
             "the same model therefore differ even if nothing else changes, and must not be compared."),
         refresher=(
             "For two independent samples of the same $d$-dimensional law the mean term contributes "
             "$\\mathbb{E}\\|\\bar x-\\bar y\\|^2=2\\,\\mathrm{Tr}\\Sigma/N$ exactly, and the covariance term adds a further "
             "$O(d^2/N)$ (Chong and Forsyth 2020, who propose extrapolating to $\\mathrm{FID}_\\infty$). The figure "
             "estimates $\\mathbb{E}[\\widehat{\\mathrm{FID}}_N]$ between two samples of one Gaussian for several $N$ and "
             "fits $c/N$; the check is the ratio between $N=500$ and $N=2000$, which $1/N$ scaling puts at 4 (the "
             "repository check uses the same tolerance window)."),
         code=r'''
dd = 8
def mean_fid(N, reps=120):
    vals = []
    for _ in range(reps):
        X, Y = rng.normal(size=(N, dd)), rng.normal(size=(N, dd))
        vals.append(fid(X.mean(0), np.cov(X.T), Y.mean(0), np.cov(Y.T)))
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(reps))
Ns = np.array([250, 500, 1000, 2000, 4000])
est = np.array([mean_fid(n) for n in Ns])
cfit = np.sum(est[:, 0] / Ns) / np.sum(1 / Ns ** 2)     # least-squares c for E[FID_N] = c / N
plt.figure(figsize=(4, 2.6)); plt.errorbar(Ns, est[:, 0], 2 * est[:, 1], fmt="o", label="simulated E[FID_N]")
plt.plot(Ns, cfit / Ns, "-", label=f"fit c/N, c = {cfit:.1f}"); plt.xscale("log"); plt.yscale("log")
plt.xlabel("samples N per set"); plt.ylabel("FID between two samples of one law"); plt.legend(); plt.tight_layout(); plt.show()
ratio = est[1, 0] / est[3, 0]
check("E[FID_500] / E[FID_2000] is close to 4", 3.4 < ratio < 4.6, f"ratio = {ratio:.2f} (1/N predicts 4.0)")
check("true FID of the two laws is 0, estimate is positive at every N", np.all(est[:, 0] > 0))
''')

    c.eq(["EQ-M-048", "EQ-S5-009"], "KID and CMMD: unbiased kernel estimators",
         r"\widehat{\mathrm{MMD}}^2_u=\frac{1}{m(m-1)}\sum_{i\ne j}\big[k(x_i,x_j)+k(y_i,y_j)\big]-\frac{2}{m(m-1)}\sum_{i\ne j}k(x_i,y_j),"
         r"\qquad k(x,y)=\Big(\frac{x^\top y}{d}+1\Big)^3\ \text{(KID)}",
         plain=(
             "Instead of fitting Gaussians, compare the two feature clouds through a kernel: average similarity within "
             "the real set, plus within the generated set, minus twice the similarity across them. Leaving out the "
             "$i=j$ terms makes the estimate unbiased, so unlike FID its expected value does not depend on the sample "
             "size; it fluctuates around the truth instead of sitting above it. KID uses a cubic polynomial kernel on "
             "Inception features; CMMD uses a Gaussian kernel on CLIP features."),
         refresher=(
             "Gretton et al. (2012) give the U-statistic above; with $k(x,y)=\\langle\\varphi(x),\\varphi(y)\\rangle$ it "
             "estimates $\\|\\mathbb{E}\\varphi(x)-\\mathbb{E}\\varphi(y)\\|^2$ without bias, which is zero iff the two laws "
             "agree when the kernel is characteristic (the cubic polynomial kernel matches moments up to order three, "
             "so KID is distribution-free only up to that order). The price is variance: KID is computed on subsets "
             "and averaged, and values are comparable only at the same kernel and subset size (the paper's benchmark "
             "uses $\\min(1000,N)$). The check draws many subsets from one law and shows the mean is zero within its "
             "standard error, whereas the biased V-statistic (keeping $i=j$) is not."),
         code=r'''
def mmd2_u(X, Y, kern):
    m = len(X); Kxx, Kyy, Kxy = kern(X, X), kern(Y, Y), kern(X, Y)
    np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0); np.fill_diagonal(Kxy, 0)
    return (Kxx.sum() + Kyy.sum()) / (m * (m - 1)) - 2 * Kxy.sum() / (m * (m - 1))
def mmd2_v(X, Y, kern):                                    # biased V-statistic, diagonal kept
    return kern(X, X).mean() + kern(Y, Y).mean() - 2 * kern(X, Y).mean()
d = 16
poly3 = lambda X, Y: (X @ Y.T / d + 1) ** 3
reps = 400; m = 200
U = np.array([mmd2_u(rng.normal(size=(m, d)), rng.normal(size=(m, d)), poly3) for _ in range(reps)])
V = np.array([mmd2_v(rng.normal(size=(m, d)), rng.normal(size=(m, d)), poly3) for _ in range(reps)])
se = U.std() / np.sqrt(reps)
check("unbiased KID has mean 0 when p = q", abs(U.mean()) < 4 * se, f"mean {U.mean():.4f}, s.e. {se:.4f}")
check("the biased V-statistic does not", V.mean() > 10 * V.std() / np.sqrt(reps), f"mean {V.mean():.4f}")
shifted = np.array([mmd2_u(rng.normal(size=(m, d)), rng.normal(0.3, 1, size=(m, d)), poly3) for _ in range(50)])
check("and it detects a mean shift", shifted.mean() > 0, f"mean {shifted.mean():.4f}")
''')

    # ------------------------------------------------------------------------------------------ coverage
    c.section("Fidelity versus coverage",
              "FID folds sample quality and sample diversity into one number. Precision and recall separate them, and "
              "density and coverage repair two failure modes of the first estimators. None of them sees copying.")

    c.eq(["EQ-S5-010"], "Precision and recall as mass in the other distribution's support",
         r"P=\Pr_{\mathbf{x}\sim p_g}\big[\mathbf{x}\in\mathrm{supp}\,p_r\big],\qquad R=\Pr_{\mathbf{x}\sim p_r}\big[\mathbf{x}\in\mathrm{supp}\,p_g\big]",
         plain=(
             "Precision asks: of the images the model produces, what fraction look like something real? Recall asks "
             "the converse: of the real images, what fraction could the model have produced? A model that only draws "
             "cats from a cats-and-dogs dataset has precision 1 and recall 1/2. Sajjadi et al. (2018) estimated both "
             "from a $k$-means histogram of the feature space."),
         refresher=(
             "Sajjadi et al. define a whole curve $(\\alpha,\\beta)$ of precision--recall pairs from the overlap of two "
             "histograms over the same partition; the support-based reading above is its limit for sharp supports. "
             "The check builds two 2-D mixtures with known supports and estimates $P$ and $R$ by a common $k$-means "
             "partition (cells with mass from both distributions count as shared support), recovering the expected "
             "values for a mode-dropping generator."),
         code=r'''
from scipy.cluster.vq import kmeans2
real = np.concatenate([rng.normal(c_, 0.15, size=(1500, 2)) for c_ in [(0, 0), (3, 0), (0, 3), (3, 3)]])
gen = np.concatenate([rng.normal(c_, 0.15, size=(3000, 2)) for c_ in [(0, 0), (3, 0)]])      # drops two of four modes
cent, _ = kmeans2(np.concatenate([real, gen]), 40, seed=0, minit="++")
lab = lambda X: np.argmin(((X[:, None, :] - cent[None]) ** 2).sum(-1), axis=1)
hr, hg = np.bincount(lab(real), minlength=40) / len(real), np.bincount(lab(gen), minlength=40) / len(gen)
shared = (hr > 0) & (hg > 0)
P_hat, R_hat = hg[shared].sum(), hr[shared].sum()
check("precision ~ 1 (every generated point lies on a real mode)", P_hat > 0.97, f"P = {P_hat:.3f}")
check("recall ~ 1/2 (two of four modes covered)", abs(R_hat - 0.5) < 0.05, f"R = {R_hat:.3f}")
''')

    c.eq(["EQ-S5-011"], "Improved precision and recall from $k$-NN manifolds",
         r"\mathrm{precision}=\frac{1}{M}\sum_{j=1}^{M}\mathbb{1}\big[\exists i:\ \|\phi_{g,j}-\phi_{r,i}\|\le\mathrm{NND}_k(\phi_{r,i})\big],"
         r"\qquad \mathrm{recall}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\big[\exists j:\ \|\phi_{r,i}-\phi_{g,j}\|\le\mathrm{NND}_k(\phi_{g,j})\big],\quad k=3",
         plain=(
             "Draw a ball around every real feature vector whose radius reaches its third-nearest real neighbour; the "
             "union of balls is a stand-in for 'the set of real-looking images'. Precision is the fraction of "
             "generated points that land inside some ball; recall swaps the roles. This is the estimator the paper's "
             "benchmark uses (on VGG-16 features, through torch-fidelity)."),
         refresher=(
             "Kynkäänniemi et al. (2019). The $k$-NN radius adapts to local density, so the estimate needs no "
             "partition; but Naeem et al. (2020) showed that a single outlier inflates its ball and that precision and "
             "recall do not reach 1 even for two samples of the same distribution. The check implements the estimator, "
             "recovers the mode-dropping case (precision near 1, recall near 1/2), and exhibits the two Naeem "
             "failures: identical distributions score below 1, and one far outlier in the real set raises precision "
             "of a generator that puts mass near it."),
         code=r'''
def knn_radius(X, k=3):
    D2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.sqrt(np.sort(D2, axis=1)[:, k])              # column 0 is the point itself
def inside(Q, X, rad):                                     # fraction of Q inside the union of balls around X
    D2 = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return float(np.mean(np.any(D2 <= rad[None, :] ** 2, axis=1)))
def improved_pr(real, gen, k=3):
    return inside(gen, real, knn_radius(real, k)), inside(real, gen, knn_radius(gen, k))
r_s, g_s = real[rng.choice(len(real), 600, replace=False)], gen[rng.choice(len(gen), 600, replace=False)]
P3, R3 = improved_pr(r_s, g_s)
check("mode dropping: precision ~ 1, recall ~ 1/2", P3 > 0.95 and abs(R3 - 0.5) < 0.08, f"P = {P3:.3f}, R = {R3:.3f}")
same_a, same_b = rng.normal(size=(600, 2)), rng.normal(size=(600, 2))
Ps, Rs = improved_pr(same_a, same_b)
check("Naeem failure 1: identical laws score below 1", Ps < 1 and Rs < 1, f"P = {Ps:.3f}, R = {Rs:.3f}")
real_out = np.concatenate([same_a, [[12.0, 12.0]]])        # one far outlier in the real set
gen_far = rng.normal(10.5, 0.5, size=(600, 2))             # a generator that is wrong everywhere except near the outlier
P_out, _ = improved_pr(real_out, gen_far)
P_no, _ = improved_pr(same_a, gen_far)
check("Naeem failure 2: one real outlier inflates precision", P_out > P_no + 0.3, f"P with outlier {P_out:.3f}, without {P_no:.3f}")
''')

    c.eq(["EQ-S5-012"], "Density and coverage",
         r"\mathrm{density}=\frac{1}{kM}\sum_{j=1}^{M}\sum_{i=1}^{N}\mathbb{1}\big[\phi_{g,j}\in B(\phi_{r,i},\mathrm{NND}_k(\phi_{r,i}))\big],"
         r"\qquad \mathrm{coverage}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\big[\exists j:\ \phi_{g,j}\in B(\phi_{r,i},\mathrm{NND}_k(\phi_{r,i}))\big]",
         plain=(
             "Density counts how many real balls each generated point falls into, divided by $k$, so that a sample of "
             "the real distribution scores about 1 and a generator that piles its mass onto a few typical real points "
             "scores above 1 instead of being capped. Coverage asks what fraction of real balls contain at least one "
             "generated point; it only uses balls around real points, so outliers in the generated set cannot "
             "inflate it."),
         refresher=(
             "Naeem et al. (2020). Density has expectation 1 under $p_g=p_r$ (each real ball contains $k$ real "
             "neighbours by construction); coverage approaches 1 as $M$ grows and is insensitive to generated outliers "
             "because only real balls are used. Both still depend on $k$ and $N$, and neither sees copying (next "
             "entry). The check shows density near 1 for identical laws, density above 1 for a generator concentrated "
             "on one real mode, and coverage of about 1/2 under mode dropping while a generated outlier leaves it "
             "unchanged."),
         code=r'''
def density_coverage(real, gen, k=3):
    rad = knn_radius(real, k)
    D2 = ((gen[:, None, :] - real[None, :, :]) ** 2).sum(-1)
    inb = D2 <= rad[None, :] ** 2                          # (M, N) generated j inside ball i
    return float(inb.sum() / (k * len(gen))), float(np.mean(np.any(inb, axis=0)))
dens_same, cov_same = density_coverage(same_a, same_b)
check("identical laws: density ~ 1", abs(dens_same - 1) < 0.2, f"density = {dens_same:.3f}, coverage = {cov_same:.3f}")
clumped = r_s[:1] + rng.normal(0, 0.005, size=(600, 2))    # all mass next to one real point of the set used
dens_cl, cov_cl = density_coverage(r_s, clumped)
check("mass piled on one real point: density > 1, coverage ~ 0", dens_cl > 1.2 and cov_cl < 0.05, f"density = {dens_cl:.2f}, coverage = {cov_cl:.3f}")
_, cov_drop = density_coverage(r_s, g_s)
_, cov_drop_out = density_coverage(r_s, np.concatenate([g_s, [[50.0, 50.0]]]))
check("mode dropping: coverage ~ 1/2, unchanged by a generated outlier", abs(cov_drop - 0.5) < 0.08 and abs(cov_drop - cov_drop_out) < 1e-9,
      f"coverage = {cov_drop:.3f} vs {cov_drop_out:.3f} with an outlier")
''')

    c.eq(["EQ-S5-017"], "Recall (and every distribution metric) is blind to copying",
         r"p_g=\hat p_{\mathrm{train}}\ \Longrightarrow\ \mathrm{FID}\approx0,\ \ \mathrm{precision}=\mathrm{recall}=\mathrm{coverage}\approx1,\ \text{although the model has generalised nothing}",
         plain=(
             "If a model simply replays its training images, its samples are, by construction, real-looking and cover "
             "every mode. Every metric that compares two distributions therefore gives it a perfect score. The "
             "paper's E5 study shows this with real models; here the point is made with toy data, and the one check "
             "that does see it (a nearest-neighbour search against the training set) is run alongside."),
         refresher=(
             "Distribution metrics are functionals of $(p_r,p_g)$ and cannot distinguish $p_g=p_r$ from "
             "$p_g=\\hat p_{\\mathrm{train}}$ when the reference set is drawn from $p_r$ and the training set is too "
             "(the empirical and population laws agree in every statistic the metric uses). Detecting memorisation "
             "needs a per-sample comparison against the training set (Carlini et al. 2023; Somepalli et al. 2023), a "
             "three-sample test (Meehan et al. 2020) or an authenticity score (Alaa et al. 2022). The check scores "
             "a 'generator' that returns the training set against a held-out reference set and against a genuine "
             "fresh sample, then runs the nearest-neighbour distance test, which separates the two."),
         code=r'''
train = rng.normal(size=(600, 2))                           # training set
reference = rng.normal(size=(600, 2))                       # held-out reference set (same law)
fresh = rng.normal(size=(600, 2))                           # a generator that generalises
copier = train.copy()                                       # a generator that memorises
f_copy = fid(copier.mean(0), np.cov(copier.T), reference.mean(0), np.cov(reference.T))
f_fresh = fid(fresh.mean(0), np.cov(fresh.T), reference.mean(0), np.cov(reference.T))
P_c, R_c = improved_pr(reference, copier); P_f, R_f = improved_pr(reference, fresh)
_, cov_c = density_coverage(reference, copier); _, cov_f = density_coverage(reference, fresh)
check("copier and generaliser get the same FID, precision, recall, coverage (within noise)",
      abs(f_copy - f_fresh) < 0.05 and abs(R_c - R_f) < 0.08 and abs(cov_c - cov_f) < 0.08,
      f"FID {f_copy:.3f} vs {f_fresh:.3f}; recall {R_c:.2f} vs {R_f:.2f}; coverage {cov_c:.2f} vs {cov_f:.2f}")
nn_to_train = lambda G: np.sqrt(((G[:, None, :] - train[None, :, :]) ** 2).sum(-1)).min(1)
check("a nearest-neighbour search against the training set separates them",
      np.median(nn_to_train(copier)) == 0 and np.median(nn_to_train(fresh)) > 0.02,
      f"median NN distance: copier {np.median(nn_to_train(copier)):.3f}, generaliser {np.median(nn_to_train(fresh)):.3f}")
''')

    # ------------------------------------------------------------------------------------------ conditional / text / cost
    c.section("Conditional, text and cost quantities")

    c.eq(["EQ-M-049", "EQ-S5-013"], "CLIPScore and its two scale conventions",
         r"\mathrm{CLIPScore}(\mathbf{x},\mathbf{c})=\kappa\cdot\max\big(\cos(E_I(\mathbf{x}),E_T(\mathbf{c})),\,0\big),\qquad \kappa\in\{2.5,\ 100\}",
         plain=(
             "Embed the image and its prompt with CLIP and take the cosine of the angle between the two vectors; "
             "negative values are clipped to zero. Hessel et al. (2021) multiply by 2.5 to spread the typical range, "
             "TorchMetrics multiplies by 100: the same cosine gives 0.75 under one convention and 30 under the other, "
             "so a reported CLIPScore means nothing without its $\\kappa$, its CLIP backbone and its prompt set."),
         refresher=(
             "The score measures semantic alignment of image and text in CLIP's embedding space, not realism, and it "
             "inherits CLIP's blind spots for relations, attribute binding and word order (Yuksekgonul et al. 2023). "
             "It is also not independent of a model guided or trained with the same encoder, which is why E5 compares "
             "it against held-out human captions and E8 against human preferences. The check verifies the factor of "
             "40 between conventions, the clipping at zero, and invariance to the embeddings' norms."),
         code=r'''
def clipscore(ei, et, kappa):
    cos = ei @ et / (np.linalg.norm(ei) * np.linalg.norm(et))
    return kappa * max(cos, 0.0)
ei, et = rng.normal(size=512), rng.normal(size=512)
et = 0.6 * ei + 0.8 * et                                    # correlated pair
s25, s100 = clipscore(ei, et, 2.5), clipscore(ei, et, 100)
check("the two conventions differ by exactly 40", *close(s100 / s25, 40.0, 1e-12))
check("negative cosine is clipped to 0", clipscore(ei, -ei, 100) == 0.0)
check("invariant to embedding norms", *close(clipscore(3 * ei, 0.1 * et, 100), s100, 1e-12))
''')

    c.eq(["EQ-S5-014"], "Bradley--Terry and Elo rankings from pairwise battles",
         r"P(i\succ j)=\sigma(\theta_i-\theta_j),\qquad \sigma(u)=\frac{1}{1+e^{-u}},\qquad \hat\theta=\arg\max_\theta\sum_{\text{battles}}\log\sigma(\theta_{\mathrm{win}}-\theta_{\mathrm{lose}})",
         plain=(
             "Show people two anonymous answers and ask which is better. Give every model a strength number "
             "$\\theta$; the model with the larger strength is more likely to win, with the probability set by the "
             "logistic curve of the difference. Fitting the strengths to the recorded wins gives a leaderboard "
             "(Chatbot Arena). Elo is an online version of the same fit."),
         refresher=(
             "Bradley and Terry (1952). Only differences $\\theta_i-\\theta_j$ are identified, so one strength is "
             "pinned; the log-likelihood is concave, so the maximum is unique and finite when every model has both won and lost along some path (the directed win graph is strongly connected; Ford, 1957). The "
             "same model is the reward model of RLHF (Chapter 11) and the judge calibration of E8. The check fits "
             "strengths to simulated battles from known strengths and recovers them up to the pinned constant, and "
             "verifies that an Elo update with a small learning rate is a stochastic-gradient step on this "
             "likelihood."),
         code=r'''
theta_true = np.array([0.0, 0.8, -0.5, 1.5])
sig = scipy.special.expit
pairs = rng.integers(0, 4, size=(6000, 2)); pairs = pairs[pairs[:, 0] != pairs[:, 1]]
wins = rng.random(len(pairs)) < sig(theta_true[pairs[:, 0]] - theta_true[pairs[:, 1]])
winner, loser = np.where(wins, pairs[:, 0], pairs[:, 1]), np.where(wins, pairs[:, 1], pairs[:, 0])
nll = lambda th: -np.sum(np.log(sig(th[winner] - th[loser])))
th0 = np.zeros(4)
res = scipy.optimize.minimize(lambda t: nll(np.concatenate([[0.0], t])), th0[1:], method="BFGS")
th_hat = np.concatenate([[0.0], res.x])
check("Bradley-Terry recovers the strengths (up to the pinned constant)", *close(th_hat, theta_true, 0.15, "max error "))
# Elo: theta_w += K (1 - sigma(d)), theta_l -= K (1 - sigma(d)) is the gradient of log sigma(theta_w - theta_l)
th = th_hat.copy(); w, l = winner[0], loser[0]; K = 1e-3
g = np.zeros(4); g[w] += 1 - sig(th[w] - th[l]); g[l] -= 1 - sig(th[w] - th[l])
dlog = lambda th_: np.log(sig(th_[w] - th_[l]))
grad_single = np.array([(dlog(th + 1e-6 * np.eye(4)[i]) - dlog(th - 1e-6 * np.eye(4)[i])) / 2e-6 for i in range(4)])
check("an Elo update is a gradient step on one battle's log-likelihood", *close(g, grad_single, 1e-6))
''')

    c.eq(["EQ-S5-015"], "The cost of a diffusion FID",
         r"\frac{\text{cost of a 50k-sample FID at }1000\text{ NFE}}{\text{cost of the same FID for a GAN}}\ \approx\ 1000\times\frac{\text{cost of one U-Net evaluation}}{\text{cost of one generator evaluation}}\ \sim\ 10^2\text{--}10^3",
         plain=(
             "A GAN makes an image in one network call; a 1000-step diffusion sampler makes a thousand. Evaluating FID "
             "on fifty thousand images therefore costs a thousand times more network calls, times the ratio of the "
             "two networks' per-call cost. That is why diffusion papers often report FID on fewer samples, and then "
             "inherit the sample-size bias shown above."),
         refresher=(
             "The statement is arithmetic on counts; the per-call cost ratio depends on the architectures. The paper's "
             "own measured budgets (Supplement S9) give forward FLOPs per sample for its DCGAN and DDPM; the check "
             "uses the paper's counts of network evaluations and confirms the order of magnitude for per-call ratios "
             "between 0.1 and 1, which spans the typical U-Net versus DCGAN generator range at $32^2$ to $64^2$ px."),
         code=r'''
n_samples, nfe_diffusion, nfe_gan = 50_000, 1000, 1
calls_ratio = (n_samples * nfe_diffusion) / (n_samples * nfe_gan)
check("illustration (arithmetic): network calls 1000x", calls_ratio == 1000)
for unet_over_gen in (0.1, 0.3, 1.0):
    total = calls_ratio * unet_over_gen
    check(f"illustration (arithmetic): per-call ratio {unet_over_gen}: total cost ratio {total:.0f}x lies in 1e2..1e3", 100 <= total <= 1000)
''')

    return c.build()


if __name__ == "__main__":
    print(build().path)
