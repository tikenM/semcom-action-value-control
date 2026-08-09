"""System backends producing receiver posteriors at operating points.

An operating point is (e, s): encoder-quality index e and channel-state index s.
Each backend returns, for a requested split, a Record with:
  y           : true labels [n]
  post_clean  : DECODER posteriors on the channel-free latent [n, K]  (for I_tx)
  post_noisy  : DECODER posteriors after the channel            [n, K]  (for I_rx)
  k           : number of transmitted latent symbols (for capacity bound)
  gamma       : per-symbol SNR, linear (for capacity bound)
Posteriors are the deployed decoder's outputs and may be miscalibrated; the
estimators calibrate them. Encoder-directed actions raise e (better/richer
encoder); channel-directed actions raise s (more power -> cleaner channel).

The ControlledBackend has closed-form Bayes posteriors, so ground-truth cause is
knowable and the analysis stack is fully testable without a GPU. The
DeepJSCCBackend mirrors the interface for full-scale runs with torch.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class Record:
    y: np.ndarray
    post_clean: np.ndarray
    post_noisy: np.ndarray
    k: int
    gamma: float


class ControlledBackend:
    """Diagonal Gaussian mixture with exact Bayes posteriors.

    Encoder quality controls class-mean separation AND latent rate k (an encoder
    upgrade transmits more symbols and separates classes more). Channel state
    controls noise variance and per-symbol SNR. A fixed over-confidence
    temperature distorts the emitted posteriors so that calibration is nontrivial.
    """

    def __init__(self, K=10, sigma_e=1.0, sep0=0.9, sep_step=0.30,
                 sc0=3.4, chan_decay=0.82, k0=4, k_step=3,
                 snr0=0.35, snr_step=1.1, overconf_T=0.6, seed=0):
        self.K = K
        self.sigma_e = sigma_e
        self.sep0, self.sep_step = sep0, sep_step
        self.sc0, self.chan_decay = sc0, chan_decay
        self.k0, self.k_step = k0, k_step
        self.snr0, self.snr_step = snr0, snr_step
        self.overconf_T = overconf_T          # <1 => decoder is overconfident
        self.mu = np.eye(K)
        self.rng = np.random.default_rng(seed)

    # -- operating-point physical parameters ------------------------------
    def sep(self, e):     return self.sep0 + self.sep_step * e
    def sigma_c(self, s): return self.sc0 * (self.chan_decay ** s)
    def k(self, e):       return int(self.k0 + self.k_step * e)
    def gamma(self, s):   return self.snr0 + self.snr_step * s   # linear SNR

    def _posterior(self, z, sep, sigma_tot, overconf=True):
        d2 = ((z[:, None, :] - sep * self.mu[None, :, :]) ** 2).sum(-1)
        logp = -d2 / (2.0 * sigma_tot ** 2)
        if overconf:
            logp = logp / self.overconf_T           # sharpen -> overconfident
        logp -= logp.max(1, keepdims=True)
        P = np.exp(logp)
        return P / P.sum(1, keepdims=True)

    def evaluate(self, e, s, n=6000, seed=None):
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        sep = self.sep(e)
        sc = self.sigma_c(s)
        sig_rx = np.sqrt(self.sigma_e ** 2 + sc ** 2)
        y = rng.integers(0, self.K, size=n)
        means = sep * self.mu[y]
        z_clean = means + rng.normal(0, self.sigma_e, size=(n, self.K))
        z_noisy = z_clean + rng.normal(0, sc, size=(n, self.K))
        post_clean = self._posterior(z_clean, sep, self.sigma_e)
        post_noisy = self._posterior(z_noisy, sep, sig_rx)
        return Record(y=y, post_clean=post_clean, post_noisy=post_noisy,
                      k=self.k(e), gamma=self.gamma(s))

    # ground-truth losses (closed form) -- used ONLY to score the method
    def true_losses(self, e, s, n=40000):
        HY = np.log2(self.K)
        rec = self.evaluate(e, s, n=n, seed=12345)
        # exact (non-overconfident) posteriors for ground truth MI
        sep, sc = self.sep(e), self.sigma_c(s)
        sig_rx = np.sqrt(self.sigma_e ** 2 + sc ** 2)
        y = rec.y
        # recompute exact posteriors without the overconfidence distortion
        means = sep * self.mu[y]
        rng = np.random.default_rng(999)
        z_clean = means + rng.normal(0, self.sigma_e, size=(n, self.K))
        z_noisy = z_clean + rng.normal(0, sc, size=(n, self.K))
        Pc = self._posterior(z_clean, sep, self.sigma_e, overconf=False)
        Pn = self._posterior(z_noisy, sep, sig_rx, overconf=False)
        I_tx = HY - np.mean(-np.sum(Pc * np.log2(Pc + 1e-12), 1))
        I_rx = HY - np.mean(-np.sum(Pn * np.log2(Pn + 1e-12), 1))
        L_enc = HY - I_tx
        L_ch = max(I_tx - I_rx, 0.0)
        bayes = 1.0 - np.mean(Pn.max(1))
        return dict(L_enc=L_enc, L_ch=L_ch, bayes=bayes)


class DeepJSCCBackend:
    """Torch DeepJSCC backend (full-scale). Same interface as ControlledBackend.

    e indexes a trained rate point (encoder upgrade = higher-rate model);
    s indexes channel SNR (power action = higher SNR). Requires torch and a
    dict of pretrained models keyed by rate. See train_deepjscc() in model.py.
    Kept import-guarded so the package runs without torch.
    """

    def __init__(self, models_by_e, channel="awgn", K=10):
        self.models = models_by_e          # {e: (encoder, decoder, k)}
        self.channel = channel
        self.K = K

    def evaluate(self, e, s, n=6000, seed=None):
        import torch  # local import; only needed in full mode
        raise NotImplementedError(
            "Wire to model.py: run encoder->channel(s)->decoder on the eval "
            "loader, return softmax posteriors (clean and noisy) and (k, gamma). "
            "The ControlledBackend validates the full analysis stack offline.")
