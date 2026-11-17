"""Chapter 11: from models to systems (main paper §6; Supplement S10)."""
from nbkit import Chapter


def build():
    c = Chapter(
        11, "systems_post_training_agents", "From models to systems: post-training, retrieval and agents",
        summary=(
            "A deployed language model is a pretrained likelihood model that has been fine-tuned on demonstrations, "
            "tilted towards a reward, wrapped in retrieval and placed in an agent loop. This chapter verifies the "
            "equations behind each stage: the pretraining and supervised fine-tuning losses are the chain rule; the "
            "KL-regularised reward objective has an exponential-tilt optimum that keeps the reference's support and "
            "cannot lower pass@$k$ under a 0/1 verifier; DPO is that tilt inverted; retrieval is a truncated "
            "latent-variable marginal; an agent's trajectory is the chain rule with an external factor; and training "
            "on one's own samples shrinks the fitted distribution. Section 8.7 of the paper (E7, E7b) measures where "
            "trained systems depart from these idealisations."),
        prerequisites="Chapters 2, 4 and 7 (maximum likelihood, the chain rule, energy-based tilts)",
        paper_sections="main paper §6; Supplement S10",
    )

    c.section("Pretraining and supervised fine-tuning are the chain rule", (
        "Both stages maximise an exact conditional likelihood over tokens; what changes is the data and the "
        "conditioning. The checks use a tiny vocabulary where every probability is an exact table entry."))

    c.eq(
        ["EQ-M-051", "EQ-S10-004"], "The supervised fine-tuning objective (a log-likelihood, maximised)",
        r"\mathcal L_{\mathrm{SFT}}(\theta)=\mathbb E_{(\mathbf c,\mathbf y)}\Big[\sum_i\log p_\theta(y_i\mid\mathbf y_{<i},\mathbf c)\Big]",
        plain=(
            "After pretraining on raw text, a model is shown pairs of a prompt $\\mathbf c$ and a good answer "
            "$\\mathbf y$ and trained to make the answer likely given the prompt, one token at a time. The formula is "
            "the pretraining loss (Chapter 4) applied to the answer tokens only, with the prompt in the context. "
            "Because it is still maximum likelihood, it still pulls the model towards covering every demonstrated "
            "answer, the property the paper calls mode-covering."),
        refresher=(
            "Ouyang et al. (2022), stage one. The objective differs from the pretraining log-likelihood of the concatenation "
            "$[\\mathbf c,\\mathbf y]$ exactly by the prompt's own log-probability $\\sum_j\\log p_\\theta(c_j|\\mathbf c_{<j})$, "
            "which is why implementations mask the prompt tokens. By Chapter 2 it minimises "
            "$\\mathbb E_{\\mathbf c}D_{\\mathrm{KL}}(p_{\\mathrm{demo}}(\\cdot|\\mathbf c)\\|p_\\theta(\\cdot|\\mathbf c))$, "
            "a forward KL per prompt, so it cannot be made to prefer one good answer over another: that is what the "
            "preference stage adds. The check builds a 2-token prompt and 2-token answer over a 3-symbol vocabulary "
            "with a random autoregressive table and verifies the masking identity."),
        code='''V = 3
logits = rng.normal(size=(4, V, V, V, V))             # position i: logits over the next token given up to 3 previous tokens (padded)
def logp_next(prefix):                                 # categorical log-probabilities for the next token given a prefix tuple
    i = len(prefix); key = (tuple(prefix) + (0, 0, 0))[:3]
    l = logits[i][key]; return l - np.logaddexp.reduce(l)
def logp_seq(seq, start=0):                            # sum over positions >= start of log p(x_i | x_<i)
    return sum(logp_next(seq[:i])[seq[i]] for i in range(start, len(seq)))
c_, y_ = [1, 2], [0, 2]
full = logp_seq(c_ + y_)                               # pretraining-style log-likelihood of the concatenation
sft = logp_seq(c_ + y_, start=len(c_))                 # SFT: answer tokens only, prompt in the context
check("SFT log-likelihood = log p([c, y]) - log p(c)", *close(sft, full - logp_seq(c_), 1e-12))
check("SFT loss is a sum of per-token conditional log-probabilities (chain rule)",
      *close(sft, logp_next(c_)[y_[0]] + logp_next(c_ + [y_[0]])[y_[1]], 1e-12))''')

    c.eq(
        "EQ-S10-003", "Scaling laws and the compute-optimal token budget (quoted constants)",
        r"L(N,D)\approx E+\frac{A}{N^\alpha}+\frac{B}{D^\beta};\qquad D_{\mathrm{opt}}/N_{\mathrm{opt}}\approx 20\ \text{tokens per parameter}",
        plain=(
            "Held-out loss falls smoothly and predictably as a model gets more parameters $N$ and more training "
            "tokens $D$: each term is a power law. Given a fixed compute budget there is a best split between size "
            "and data, and the widely used rule of thumb from the Chinchilla study is about twenty tokens per "
            "parameter. These are empirical fits from the cited papers, not theorems; the paper quotes them, and the "
            "notebook checks only what follows from the quoted numbers. One thing that follows is a warning: the "
            "published parametric fit does *not* reproduce the twenty-tokens rule, which came from the same paper's "
            "other two estimation methods."),
        refresher=(
            "Kaplan et al. (2020) fit power laws in $N$, $D$ and compute separately; Hoffmann et al. (2022) fit the "
            "joint form above (Approach 3: $E=1.69$, $A=406.4$, $B=410.7$, $\\alpha=0.34$, $\\beta=0.28$) and, from "
            "isoFLOP profiles (Approaches 1--2), derive $N_{\\mathrm{opt}},D_{\\mathrm{opt}}\\propto C^{1/2}$ with "
            "about 20 tokens per parameter at $C=6ND$. Minimising the Approach-3 formula under $6ND=C$ gives "
            "exponents $a\\approx0.46$, $b\\approx0.54$ (their Table 2) and a ratio that grows from about 30 at "
            "$10^{19}$ FLOPs to about 100 at $10^{24}$, inconsistent with the headline rule; Besiroglu et al. (2024) "
            "trace this to the reported fit and obtain $\\approx20$ after refitting. The constants depend on the "
            "tokeniser and data and have been refit since; E6b in Section 8.7 uses $C=6ND$ on the Pythia suite and "
            "finds the compute-matched ordering of sizes differs from the token-matched one. The checks verify the "
            "arithmetic of the rule, the Approach-3 exponents, and that the Approach-3 ratio is *not* 20, so that a "
            "reader does not take the quoted constants and the quoted rule to be the same statement."),
        code='''E_, A_, B_, al, be = 1.69, 406.4, 410.7, 0.34, 0.28          # Hoffmann et al. (2022), Approach 3, quoted
L = lambda N, D: E_ + A_ / N**al + B_ / D**be
Cs = np.logspace(19, 24, 6); Nopt = []
for C in Cs:
    res = scipy.optimize.minimize_scalar(lambda logN: L(math.exp(logN), C / (6 * math.exp(logN))), bounds=(10, 40), method="bounded")
    Nopt.append(math.exp(res.x))
Nopt = np.array(Nopt); Dopt = Cs / (6 * Nopt); ratio = Dopt / Nopt
a_exp = np.polyfit(np.log(Cs), np.log(Nopt), 1)[0]; b_exp = np.polyfit(np.log(Cs), np.log(Dopt), 1)[0]
check("Approach-3 constants: N_opt ~ C^0.46 and D_opt ~ C^0.54 (Hoffmann et al., Table 2)", abs(a_exp - 0.46) < 0.02 and abs(b_exp - 0.54) < 0.02,
      f"fitted exponents a = {a_exp:.3f}, b = {b_exp:.3f}")
check("Approach-3 constants do NOT give 20 tokens/parameter: the ratio runs from ~30 to ~100 over 1e19-1e24 FLOPs",
      ratio.min() > 25 and ratio.max() > 80, f"ratios {np.round(ratio, 0)}")
check("illustration (arithmetic): 70B parameters x 20 = 1.4T tokens", 20 * 70e9 == 1.4e12)
check("at the compute-optimal point the two power-law terms are of the same order (why both N and D must grow)",
      0.3 < (A_ / Nopt[2]**al) / (B_ / Dopt[2]**be) < 3, f"A/N^alpha : B/D^beta = {(A_ / Nopt[2]**al) / (B_ / Dopt[2]**be):.2f} at C = 1e21")''')

    c.section("Preference optimisation: a learned reward in place of a likelihood")

    c.eq(
        ["EQ-M-052", "EQ-S10-005"], "The Bradley--Terry reward model",
        r"p(\mathbf y_w\succ\mathbf y_l\mid\mathbf c)=\sigma\big(r(\mathbf c,\mathbf y_w)-r(\mathbf c,\mathbf y_l)\big),\qquad "
        r"\mathcal L_{\mathrm{RM}}(\psi)=-\mathbb E\big[\log\sigma\big(r_\psi(\mathbf c,\mathbf y_w)-r_\psi(\mathbf c,\mathbf y_l)\big)\big]",
        plain=(
            "People find it easier to say which of two answers is better than to score one answer. The "
            "Bradley--Terry model turns such comparisons into numbers: each answer has a hidden score $r$, and the "
            "probability that the first answer wins is the logistic function of the score difference. A reward model "
            "is a network trained to output those scores so that the observed preferences are likely. Only "
            "differences of scores matter, so the scores are defined up to an additive constant per prompt."),
        refresher=(
            "Bradley and Terry (1952); applied to RLHF by Christiano et al. (2017) and Stiennon et al. (2020). The "
            "likelihood is concave in the scores, identifies them up to a per-prompt constant when the comparison "
            "graph is connected, and is the maximum-entropy model given expected pairwise wins. The reward is "
            "trained on comparisons between outputs, never on outputs versus data, so it exerts no pressure towards "
            "the data distribution: the paper's third reason RLHF differs from a GAN. The check fits scores by "
            "minimising $\\mathcal L_{\\mathrm{RM}}$ on the population of all pairs of six answers with known true "
            "scores and recovers them up to a constant."),
        code='''sig = lambda u: 1 / (1 + np.exp(-u))
r_true = rng.normal(size=6)
P = sig(r_true[:, None] - r_true[None, :])                     # population preference matrix
def L_rm(r):
    u = r[:, None] - r[None, :]
    return -np.sum(P * np.log(sig(u)))                           # expected log-loss over all ordered pairs
res = scipy.optimize.minimize(L_rm, np.zeros(6), method="BFGS", options={"gtol": 1e-10})
r_hat = res.x - res.x.mean()
check("Bradley-Terry log-loss recovers the true scores up to an additive constant", *close(r_hat, r_true - r_true.mean(), 1e-4))
check("illustration (identity): sigma(u) + sigma(-u) = 1, so the two orderings of a pair are complementary", *close(P + P.T, 1.0, 1e-14))''')

    c.eq(
        ["EQ-M-053", "EQ-S10-006", "EQ-S10-007"], "The KL-regularised objective and its exponential-tilt optimum",
        r"\max_\theta\;\mathbb E_{\mathbf c,\,\mathbf y\sim\pi_\theta}[r(\mathbf c,\mathbf y)]-\beta\,\mathbb E_{\mathbf c}\big[D_{\mathrm{KL}}(\pi_\theta(\cdot|\mathbf c)\,\|\,\pi_{\mathrm{ref}}(\cdot|\mathbf c))\big];\qquad "
        r"\pi^*(\mathbf y|\mathbf c)=\frac{\pi_{\mathrm{ref}}(\mathbf y|\mathbf c)\exp\big(r(\mathbf c,\mathbf y)/\beta\big)}{Z(\mathbf c)}",
        plain=(
            "The policy is asked to get high reward while staying close (in KL) to the reference model it started "
            "from; $\\beta$ sets the price of moving away. If the policy could be any distribution at all, the best "
            "one has a simple form: take the reference distribution and multiply each answer's probability by "
            "$e^{r/\\beta}$, then renormalise. Good answers are boosted, bad ones suppressed, but nothing the "
            "reference considered impossible becomes possible. The paper calls this an energy-based *tilt* of the "
            "reference (Chapter 7) and builds its reading of post-training on it."),
        refresher=(
            "Korbak et al. (2022); Rafailov et al. (2023), Appendix A. Write the objective per prompt as "
            "$-\\beta\\,D_{\\mathrm{KL}}(\\pi\\|\\pi_{\\mathrm{ref}}e^{r/\\beta}/Z)+\\beta\\log Z$; the KL term is "
            "minimised (to zero) at the tilt, so the optimal value is $\\beta\\log Z(\\mathbf c)$. The closed form holds "
            "over all conditional distributions and requires $Z(\\mathbf c)=\\mathbb E_{\\pi_{\\mathrm{ref}}}[e^{r/\\beta}]<\\infty$; "
            "a parametric policy trained by PPO or GRPO approximates it, and E7 in Section 8.7 measures how far "
            "trained policies land from the exact tilt with the same success rate (they spend many times the KL the "
            "tilt needs). The check (ported from `rlhf_kl_regularised_optimum_is_exponential_tilt`) maximises the "
            "objective numerically over the simplex and compares with the closed form and its optimal value."),
        code='''def kl_disc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float); m = p > 0
    return math.inf if np.any(q[m] == 0) else float(np.sum(p[m] * np.log(p[m] / q[m])))
n = 8
pref = rng.dirichlet(np.ones(n)); r = rng.normal(size=n); beta = 0.7
obj = lambda pi: pi @ r - beta * kl_disc(pi, pref)
sm = lambda l: np.exp(l - l.max()) / np.exp(l - l.max()).sum()
res = scipy.optimize.minimize(lambda l: -obj(sm(l)), np.zeros(n), method="BFGS", options={"gtol": 1e-12})
pistar = pref * np.exp(r / beta); Z = pistar.sum(); pistar /= Z
check("numerical argmax over the simplex = pi_ref exp(r/beta) / Z", *close(sm(res.x), pistar, 1e-5))
check("optimal value = beta log Z", *close(obj(pistar), beta * math.log(Z), 1e-12))''')

    c.eq(
        ["EQ-M-054", "EQ-M-056", "EQ-S10-011"], "The tilt keeps the support and concentrates as $\\beta\\to0$",
        r"\mathrm{supp}\,\pi^*(\cdot|\mathbf c)=\mathrm{supp}\,\pi_{\mathrm{ref}}(\cdot|\mathbf c);\qquad "
        r"\lim_{\beta\to0}\pi^*(\cdot|\mathbf c)=\pi_{\mathrm{ref}}\big(\cdot\,\big|\,\mathbf c,\ \mathbf y\in\arg\max r(\mathbf c,\cdot)\big)",
        plain=(
            "Because the optimum multiplies the reference, it can move probability around but cannot create it "
            "where the reference had none: post-training in this idealised form never invents new answers, it "
            "re-weights existing ones. As $\\beta$ shrinks (the KL price goes to zero), all the weight piles onto the "
            "answers with the highest reward. If the reward model is wrong about which answers are best, that is "
            "where the policy ends up, which is what the paper means by over-optimisation of an imperfect reward."),
        refresher=(
            "Immediate from the closed form: $\\pi^*>0$ iff $\\pi_{\\mathrm{ref}}>0$, and for $\\beta\\to0$ the "
            "tilt $e^{r/\\beta}$ is dominated by $\\max r$, giving the reference distribution conditioned on the "
            "arg-max set. Softmax policies give every sequence non-zero mass, so in practice 'support' means "
            "non-negligible mass, and the empirical question is how mass moves; E7 counts problems whose success "
            "probability falls to zero under GRPO without the KL anchor. Gao et al. (2023) measure gold-reward loss "
            "growing with KL distance, the over-optimisation that the $\\beta\\to0$ limit predicts for an imperfect "
            "$r$. The check verifies both limits numerically."),
        code='''check("support preserved: pi* > 0 exactly where pi_ref > 0", np.all((pistar > 0) == (pref > 0)))
pref_sparse = pref.copy(); pref_sparse[[1, 5]] = 0; pref_sparse /= pref_sparse.sum()
tilt_sparse = pref_sparse * np.exp(r / beta); tilt_sparse /= tilt_sparse.sum()
check("answers with zero reference mass stay at zero whatever the reward", np.all(tilt_sparse[[1, 5]] == 0))
ls = np.log(pref) + r / 1e-3; small = np.exp(ls - np.logaddexp.reduce(ls))
check("beta -> 0: mass concentrates on argmax r", small[np.argmax(r)] > 0.999, f"mass on argmax r = {small[np.argmax(r)]:.5f}")
# with an imperfect reward, the beta -> 0 limit lands on the reward's error
r_wrong = r.copy(); r_wrong[np.argmin(r)] = r.max() + 1          # the reward model mistakenly loves the worst answer
ls = np.log(pref) + r_wrong / 1e-3; small = np.exp(ls - np.logaddexp.reduce(ls))
check("an imperfect reward is over-optimised: beta -> 0 puts the mass on the reward's error", small[np.argmin(r)] > 0.999)''')

    c.eq(
        ["EQ-M-055", "EQ-S10-008", "EQ-S10-009"], "Direct preference optimisation: the tilt inverted",
        r"r(\mathbf c,\mathbf y)=\beta\log\frac{\pi^*(\mathbf y|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf y|\mathbf c)}+\beta\log Z(\mathbf c);\qquad "
        r"\mathcal L_{\mathrm{DPO}}(\theta)=-\mathbb E\Big[\log\sigma\Big(\beta\log\tfrac{\pi_\theta(\mathbf y_w|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf y_w|\mathbf c)}"
        r"-\beta\log\tfrac{\pi_\theta(\mathbf y_l|\mathbf c)}{\pi_{\mathrm{ref}}(\mathbf y_l|\mathbf c)}\Big)\Big]",
        plain=(
            "Solve the tilt formula for the reward: the reward is $\\beta$ times the log-ratio of the optimal policy "
            "to the reference, plus a per-prompt constant. Put that into the Bradley--Terry formula and the constant "
            "cancels, because both answers share the prompt. What remains is a loss written purely in terms of the "
            "policy and the reference, with no reward network and no sampling loop: the policy is trained directly "
            "to make preferred answers relatively more likely than rejected ones."),
        refresher=(
            "Rafailov et al. (2023). The inversion is an identity only for the unconstrained optimum $\\pi^*$; "
            "DPO substitutes the parametric $\\pi_\\theta$, so the anchor to $\\pi_{\\mathrm{ref}}$ is only the "
            "log-ratio inside the logit. Azar et al. (2024) show that with deterministic preferences the loss is "
            "minimised by driving $\\pi_\\theta(\\mathbf y_l)$ to zero for any $\\beta$, and E7 finds DPO far from the "
            "tilt in KL. The checks (ported from `dpo_implicit_reward_identity_and_loss`): the inversion recovers "
            "$r$; the Bradley--Terry probability computed from implicit rewards equals the true one; and the "
            "population DPO loss over all pairs is minimised at $\\pi^*$."),
        code='''check("beta log(pi*/pi_ref) + beta log Z = r", *close(beta * np.log(pistar / pref) + beta * math.log(Z), r, 1e-12))
w, l = 1, 4
bt = sig(r[w] - r[l])
dpo_logit = beta * math.log(pistar[w] / pref[w]) - beta * math.log(pistar[l] / pref[l])
check("Bradley-Terry from implicit rewards: log Z cancels", *close(sig(dpo_logit), bt, 1e-12))
Pbt = sig(r[:, None] - r[None, :])
def dpo_loss(logits):
    lp = logits - np.logaddexp.reduce(logits)
    h = beta * (lp - np.log(pref)); u = h[:, None] - h[None, :]
    return -np.sum(Pbt * np.log(sig(u)))
res = scipy.optimize.minimize(dpo_loss, np.zeros(n), method="BFGS", options={"gtol": 1e-10})
check("argmin of the population DPO loss = pi*", *close(np.exp(res.x - np.logaddexp.reduce(res.x)), pistar, 1e-4))''')

    c.eq(
        "EQ-S10-010", "Verifiable rewards and group-relative advantages",
        r"\hat A_i=\frac{r_i-\mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G})},\qquad i=1,\dots,G\ \text{samples for one prompt}",
        plain=(
            "When the reward can be computed by a program (does the answer match? do the tests pass?), no reward "
            "network is needed. GRPO draws a group of $G$ answers to the same prompt, scores each, and uses as the "
            "learning signal how much better or worse each answer is than its group-mates, in units of the group's "
            "spread. If every answer in the group gets the same score, there is nothing to learn from that prompt."),
        refresher=(
            "Shao et al. (2024); used with or without a KL term by DeepSeek-R1, DAPO and Tülu 3, which differ on "
            "the anchor. The normalisation is a per-prompt baseline (variance reduction) and a per-prompt scale, so "
            "the update is invariant to positive affine changes of the reward and prompts with all-correct or all-wrong "
            "groups contribute zero gradient, which concentrates learning on prompts of intermediate difficulty. "
            "The main paper (§6) notes that recipes with a verifier replace the learned reward's error by the "
            "verifier's. The checks verify mean zero and unit variance, affine invariance, and the zero signal of a "
            "uniform group."),
        code='''rs = rng.integers(0, 2, size=8).astype(float); rs[0] = 1; rs[1] = 0      # 0/1 rewards for G = 8 samples
A = (rs - rs.mean()) / rs.std()
check("group-relative advantages have mean 0 and standard deviation 1", abs(A.mean()) < 1e-12 and abs(A.std() - 1) < 1e-12)
A2 = ((3 * rs + 5) - (3 * rs + 5).mean()) / (3 * rs + 5).std()
check("invariant to a positive affine change of the reward", *close(A, A2, 1e-12))
same = np.ones(8)
A_same = (same - same.mean()) / (same.std() + 1e-8)       # std = 0: implementations add a small epsilon
check("an all-correct (or all-wrong) group gives zero advantage (with epsilon in the std): no signal", np.all(A_same == 0))''')

    c.eq(
        "EQ-M-059", "The exact tilt cannot lower pass@$k$ under a 0/1 verifier",
        r"p^*=\frac{p\,e^{1/\beta}}{p\,e^{1/\beta}+1-p}\ \ge\ p\quad\Rightarrow\quad "
        r"\mathrm{pass@}k(\pi^*)=1-(1-p^*)^k\ \ge\ 1-(1-p)^k=\mathrm{pass@}k(\pi_{\mathrm{ref}})\ \ \forall k",
        plain=(
            "Suppose the reward is 1 for a correct answer and 0 otherwise, and a prompt's chance of a correct answer "
            "under the base model is $p$. The exact tilt multiplies the correct answers by $e^{1/\\beta}$ and the "
            "wrong ones by 1, so the chance of a correct answer can only go up. 'pass@$k$' is the chance that at "
            "least one of $k$ tries is correct; it goes up with $p$, so the tilted policy is at least as good as the "
            "base for every $k$. Reports that the base model overtakes a trained one at large $k$ therefore say "
            "that the trained policy is *not* the exact tilt, which is exactly what E7 measures."),
        refresher=(
            "The registry marks this entry as not verified in the October version because the earlier wording "
            "treated the crossover as *predicted* by the tilt; the present text (§6 and §8.7) states the opposite, "
            "proved here: with $r\\in\\{0,1\\}$ the tilt is the two-point formula above and "
            "$\\mathrm{pass@}k$ is monotone in $p$. The unbiased pass@$k$ estimator of Chen et al. (2021), "
            "$1-\\binom{n-c}{k}/\\binom{n}{k}$ from $c$ correct out of $n$ samples, is also checked against the exact "
            "value. E7 (Section 8.7) finds every trained policy below the tilt of equal pass@1 at $k=16$ and GRPO "
            "without the anchor losing problems the base solves; the reports of Yue et al. (2025) describe trained "
            "policies, not the tilt. Ported from `rlhf_tilt_with_binary_verifier_cannot_lower_pass_at_k` and "
            "extended with the analytic form."),
        code='''pass_k = lambda p, k: 1 - (1 - p) ** k
worst = np.inf
for _ in range(500):
    pr = rng.dirichlet(np.ones(10) * 0.3); rr = (rng.random(10) < 0.3).astype(float)
    for b in (0.05, 0.5, 5.0):
        pi = pr * np.exp(rr / b); pi /= pi.sum()
        p0, p1 = pr @ rr, pi @ rr
        pa = p0 * math.exp(1 / b) / (p0 * math.exp(1 / b) + 1 - p0)         # analytic two-point tilt
        assert abs(pa - p1) < 1e-12
        for k in (1, 4, 16, 64, 256):
            worst = min(worst, pass_k(p1, k) - pass_k(p0, k))
check("pass@k(pi*) - pass@k(pi_ref) >= 0 over 500 random prompts, three betas, five k", worst >= -1e-12, f"minimum gap {worst:.2e}")
# unbiased estimator of pass@k from n samples with c correct (Chen et al. 2021)
def pass_at_k_hat(nn, cc, k):
    return 1.0 if nn - cc < k else 1 - np.prod(1 - k / np.arange(nn - cc + 1, nn + 1))
p = 0.3; nn = 256; ks = [1, 16, 256]
est = np.array([[pass_at_k_hat(nn, rng.binomial(nn, p), k) for k in ks] for _ in range(4000)]).mean(0)
check("unbiased estimator averages to the exact pass@k", *close(est, [pass_k(p, k) for k in ks], 0.01))''')

    c.section("Retrieval, agents and self-generated data")

    c.eq(
        ["EQ-M-057", "EQ-S10-012", "EQ-S10-013", "EQ-S10-014"], "Retrieval-augmented generation as a truncated latent-variable marginal",
        r"p_\xi(\mathbf z|\mathbf c)\propto\exp\big(f(\mathbf c)^\top g(\mathbf z)\big);\qquad "
        r"p_\theta(\mathbf y|\mathbf c)\approx\sum_{\mathbf z\in\mathrm{top}\text{-}k(\mathbf c)}\tilde p_\xi(\mathbf z|\mathbf c)\,p_\theta(\mathbf y|\mathbf c,\mathbf z);\qquad "
        r"\text{RAG-Token: }\prod_i\sum_{\mathbf z}\tilde p_\xi(\mathbf z|\mathbf c)\,p_\theta(y_i|\mathbf c,\mathbf z,\mathbf y_{<i})",
        plain=(
            "A retriever scores every passage in a library against the question by a dot product of embeddings and "
            "turns the scores into probabilities. The generator then answers conditioned on a passage, and the "
            "model's overall probability of an answer is the average over passages, weighted by the retriever. In "
            "practice only the top $k$ passages are kept and their weights renormalised. RAG-Sequence averages "
            "whole answers; RAG-Token averages at every token, which is a different distribution. The paper reads "
            "this as a latent-variable model (Chapter 3) in which the retriever plays the role of the prior, and the sum over passages is cut short to the top $k$."),
        refresher=(
            "Lewis et al. (2020), with the dense retriever of Karpukhin et al. (2020). Truncation to the top $k$ "
            "without renormalisation leaves a sub-probability; with renormalisation it is a proper distribution "
            "over a prompt-dependent support. Gradients with respect to the retriever flow only through the "
            "renormalised scores of retrieved passages, never to passages outside the top $k$ or to a frozen index "
            "(EQ-S10-014), which is why REALM refreshes its index asynchronously (the main paper quotes the exact-match "
            "drop from a stale index). The checks (ported from `rag_sequence_vs_token_marginals`): both forms are "
            "normalised but differ; truncation without renormalisation is sub-normalised; and by autograd the "
            "gradient of the truncated marginal with respect to a non-retrieved passage's score is zero."),
        code='''Zn, Vv, Lq = 4, 3, 3
scores = torch.randn(Zn, dtype=torch.float64, requires_grad=True)        # f(c)^T g(z) for each passage
pz = torch.softmax(scores, 0)
py = torch.tensor(rng.dirichlet(np.ones(Vv), size=(Zn, Lq)))              # p(y_i | c, z)
import itertools
ys = list(itertools.product(range(Vv), repeat=Lq))
seq = {y: sum(pz[z] * torch.prod(torch.stack([py[z, i, y[i]] for i in range(Lq)])) for z in range(Zn)) for y in ys}
tok = {y: torch.prod(torch.stack([sum(pz[z] * py[z, i, y[i]] for z in range(Zn)) for i in range(Lq)])) for y in ys}
check("RAG-Sequence and RAG-Token are both normalised", abs(sum(seq.values()).item() - 1) < 1e-12 and abs(sum(tok.values()).item() - 1) < 1e-12)
check("...but they are different distributions", max(abs(seq[y] - tok[y]).item() for y in ys) > 1e-3)
k = 2; top = torch.argsort(-pz.detach())[:k]
trunc = sum(sum(pz[z] * torch.prod(torch.stack([py[z, i, y[i]] for i in range(Lq)])) for z in top) for y in ys)
ren = pz[top] / pz[top].sum()
trunc_r = sum(sum(ren[j] * torch.prod(torch.stack([py[z, i, y[i]] for i in range(Lq)])) for j, z in enumerate(top)) for y in ys)
check("top-k truncation: sub-normalised without renormalisation, normalised with it", trunc.item() < 1 - 1e-6 and abs(trunc_r.item() - 1) < 1e-12,
      f"{trunc.item():.3f} vs {trunc_r.item():.3f}")
# differentiability: the renormalised truncated marginal of one answer has zero gradient w.r.t. non-retrieved scores
y0 = ys[5]
p_y0 = sum(ren[j] * torch.prod(torch.stack([py[z, i, y0[i]] for i in range(Lq)])) for j, z in enumerate(top))
grad, = torch.autograd.grad(p_y0, scores)
outside = [z for z in range(Zn) if z not in top.tolist()]
check("gradient reaches only the retrieved passages' scores", torch.all(grad[outside].abs() < 1e-12) and torch.any(grad[top].abs() > 1e-6),
      f"|grad| outside top-k = {grad[outside].abs().max().item():.1e} (round-off), inside = {grad[top].abs().max().item():.2e}")''')

    c.eq(
        ["EQ-M-058", "EQ-S10-015"], "An agent's trajectory is the chain rule with an external factor",
        r"p_\theta(a_{1:H},o_{1:H}\mid\mathbf c)=\prod_{t=1}^{H}\pi_\theta(a_t\mid\mathbf c,h_t)\,\mathcal T(o_t\mid\mathbf c,h_t,a_t),"
        r"\qquad h_t=(a_1,o_1,\dots,a_{t-1},o_{t-1})",
        plain=(
            "An agent alternates between acting (emitting text or a tool call, drawn from the language model) and "
            "observing (what the tool or the world returns, drawn from the environment $\\mathcal T$). The "
            "probability of a whole episode is the product of these alternating factors, exactly as the probability "
            "of a sentence is the product of next-token probabilities. The difference is that every other factor "
            "belongs to the environment, not to the model: changing the tools changes what the agent does even "
            "when the weights are frozen."),
        refresher=(
            "The factorisation is the chain rule applied to the interleaved sequence, with $\\mathcal T$ external and "
            "non-differentiable; it is a description of inference-time interaction, not a training objective, and "
            "most deployed agents never optimise it (the paper's §6). Two consequences the paper draws: network "
            "evaluations per sample no longer bound cost (the environment and the scaffold set it), and success "
            "depends on $\\mathcal T$ and the scaffold as much as on $\\theta$, which E7b quantifies (the scaffold "
            "explains a third to a half of the variance in success). The checks build a two-step toy agent, verify "
            "the product normalises to one over all trajectories, and show that changing $\\mathcal T$ with "
            "$\\theta$ fixed changes the trajectory distribution."),
        code='''nA, nO, H = 2, 2, 2
pi = rng.dirichlet(np.ones(nA), size=(1 + nA * nO + (nA * nO) ** 2,))   # pi(a | history index), enough rows for all histories
def T_env(kind):
    return rng.dirichlet(np.ones(nO), size=(nA,)) if kind == 0 else np.eye(nO)[::-1][:nA]   # two environments
def traj_dist(Tm):
    out = {}
    for a1 in range(nA):
        for o1 in range(nO):
            for a2 in range(nA):
                for o2 in range(nO):
                    h2 = 1 + a1 * nO + o1
                    out[(a1, o1, a2, o2)] = pi[0, a1] * Tm[a1, o1] * pi[h2, a2] * Tm[a2, o2]
    return out
d0, d1 = traj_dist(T_env(0)), traj_dist(T_env(1))
check("the product of policy and environment factors sums to one over all trajectories", abs(sum(d0.values()) - 1) < 1e-12 and abs(sum(d1.values()) - 1) < 1e-12)
check("changing the environment with the weights fixed changes the trajectory distribution", max(abs(d0[k_] - d1[k_]) for k_ in d0) > 0.05)''')

    c.eq(
        "EQ-S10-016", "Model collapse: refitting on your own samples shrinks the variance",
        r"\hat\sigma^2_{k+1}=\hat\sigma^2_k\cdot\frac{\chi^2_{n-1}}{n}\quad\Rightarrow\quad "
        r"\mathbb E[\hat\sigma^2_{k+1}]=\Big(1-\tfrac1n\Big)\mathbb E[\hat\sigma^2_k]\ \xrightarrow[k\to\infty]{}\ 0",
        plain=(
            "Fit a Gaussian to $n$ samples, draw $n$ new samples from the fit, fit again, and repeat. Each round the "
            "estimated variance is the previous one times a random factor whose average is slightly less than one, "
            "so generation after generation the fitted distribution narrows until it collapses to a point. "
            "This is the paper's model-level picture of what happens when generated data replace real "
            "data in training; it is avoided when real data are kept and accumulated."),
        refresher=(
            "Shumailov et al. (2024); the Gaussian recursion is their worked example: with the maximum-likelihood "
            "variance $\\hat\\sigma^2=\\tfrac1n\\sum(x_i-\\bar x)^2$, $n\\hat\\sigma^2_{k+1}/\\hat\\sigma^2_k\\sim\\chi^2_{n-1}$, "
            "so the expectation decays geometrically and $\\log\\hat\\sigma^2_k$ is a random walk with negative "
            "drift. Gerstgrasser et al. (2024) show that accumulating rather than replacing data bounds the loss; "
            "Dohmatob et al. (2025) show that even a fixed synthetic fraction can stop scaling. The main paper reads "
            "the mechanism as a tail-truncated maximum-likelihood fit. The check simulates the recursion and compares "
            "the mean variance after $k$ rounds with $(1-1/n)^k$."),
        code='''n_, K_, R_ = 20, 30, 4000
v = np.ones(R_)
means = [v.mean()]
for k in range(K_):
    x = rng.normal(size=(R_, n_)) * np.sqrt(v)[:, None]         # draw n samples from the current fit (mean 0 w.l.o.g.)
    v = x.var(axis=1)                                            # refit the ML variance
    means.append(v.mean())
expected = (1 - 1 / n_) ** np.arange(K_ + 1)
check("E[variance after k rounds] = (1 - 1/n)^k", *close(means, expected, 0.03))
check("the fitted variance collapses: after 30 rounds the median is far below 1", np.median(v) < 0.3, f"median {np.median(v):.3f}, mean {v.mean():.3f}")''')

    c.eq(
        "EQ-S10-017", "A student trained on a channel's output converges to the channel, not to the data",
        r"\arg\min_\theta D_{\mathrm{KL}}(q_{\mathrm{syn}}\,\|\,p_\theta)=q_{\mathrm{syn}}\quad\text{when }q_{\mathrm{syn}}\text{ is in the model family}",
        plain=(
            "Maximum likelihood fits whatever distribution produced the training data (Chapter 2). If the training "
            "data came from another model, a 'channel' that already distorted the real data, the student converges "
            "to that channel's distribution, distortions included, and never to the real data. Distillation uses this "
            "on purpose; model collapse is the same fact working against you."),
        refresher=(
            "A restatement of EQ-M-003 with $q_{\\mathrm{syn}}$ in place of $p_{\\mathrm{data}}$: the forward KL is "
            "zero iff $p_\\theta=q_{\\mathrm{syn}}$, attainable when the family contains $q_{\\mathrm{syn}}$, and "
            "the finite-sample MLE converges to it. The supplement uses it to say that filtering generated data by an "
            "external verifier changes $q_{\\mathrm{syn}}$ and so changes what the student learns. The check fits a "
            "categorical model to samples from a distorted copy of the data and shows it approaches the copy, not "
            "the data."),
        code='''p_data = rng.dirichlet(np.ones(6))
q_syn = p_data ** 2; q_syn /= q_syn.sum()                      # a channel that sharpens the data distribution
sample = rng.choice(6, size=200_000, p=q_syn)
p_hat = np.bincount(sample, minlength=6) / sample.size         # categorical MLE on the channel's output
check("student MLE is close to the channel q_syn", *close(p_hat, q_syn, 0.005))
check("...and not to the data p_data", np.max(np.abs(p_hat - p_data)) > 0.05, f"max |p_hat - p_data| = {np.max(np.abs(p_hat - p_data)):.3f}")''')

    return c.build()


if __name__ == "__main__":
    print(build().path)
