"""Track A: hand-crafted microstructure features.

Each feature targets one latent variable of the synthetic generator described in
the reference PDF: grain size, second-phase fraction, porosity, grain elongation /
alignment, size uniformity, plus imaging nuisance (blur, noise, contrast) so the
regressor can de-bias them.  All orientation features are direction-invariant.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")
import cv2
from skimage import filters, measure, morphology, segmentation, feature
from skimage.feature import graycomatrix, graycoprops
from sklearn.mixture import GaussianMixture


# ----------------------------------------------------------------------------- helpers
def _stretch(img: np.ndarray, lo=1, hi=99) -> np.ndarray:
    a, b = np.percentile(img, [lo, hi])
    if b - a < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    return np.clip((img - a) / (b - a), 0, 1).astype(np.float32)


def _noise_std(img01: np.ndarray) -> float:
    """Robust noise estimate: MAD of the residual after a 3x3 median."""
    med = cv2.medianBlur((img01 * 255).astype(np.uint8), 3).astype(np.float32) / 255
    r = img01 - med
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def _radial_profile(power: np.ndarray, nbins=32):
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices(power.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    rmax = min(cy, cx)
    bins = np.linspace(0, rmax, nbins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    prof = np.zeros(nbins)
    flat = power.ravel()
    for i in range(nbins):
        m = idx == i
        prof[i] = flat[m].mean() if m.any() else 0
    return prof


def _circ_stats(theta, w=None):
    """Second-moment circular statistic for axial (0..pi) data -> alignment in [0,1]."""
    theta = np.asarray(theta, dtype=float)
    if len(theta) == 0:
        return 0.0
    if w is None:
        w = np.ones_like(theta)
    w = np.asarray(w, dtype=float)
    w = w / (w.sum() + 1e-9)
    c = (w * np.cos(2 * theta)).sum()
    s = (w * np.sin(2 * theta)).sum()
    return float(np.hypot(c, s))


def _boundary_label(den01: np.ndarray):
    """Canny boundaries -> connected non-boundary regions (grains)."""
    u8 = (den01 * 255).astype(np.uint8)
    med = float(np.median(u8))
    lo, hi = int(max(0, 0.66 * med)), int(min(255, 1.33 * med))
    edges = cv2.Canny(u8, lo, hi)
    bnd = morphology.dilation(edges > 0, morphology.disk(1))
    lab = measure.label(~bnd, connectivity=1)
    return edges, bnd, lab


# ----------------------------------------------------------------------------- feature groups
def quality_features(raw01: np.ndarray, den01: np.ndarray) -> dict:
    f = {}
    f["q_mean"] = float(raw01.mean())
    f["q_std"] = float(raw01.std())
    p = np.percentile(raw01, [1, 5, 25, 50, 75, 95, 99])
    f["q_p1"], f["q_p5"], f["q_p25"], f["q_p50"], f["q_p75"], f["q_p95"], f["q_p99"] = map(float, p)
    f["q_range"] = float(p[-1] - p[0])
    f["q_noise"] = _noise_std(raw01)
    f["q_lapvar_raw"] = float(cv2.Laplacian(raw01, cv2.CV_32F).var())
    f["q_lapvar_den"] = float(cv2.Laplacian(den01, cv2.CV_32F).var())
    gx = cv2.Sobel(den01, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(den01, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    f["q_grad_mean"] = float(mag.mean())
    f["q_grad_p90"] = float(np.percentile(mag, 90))
    f["q_grad_std"] = float(mag.std())
    sd = den01.std() + 1e-9
    f["q_skew"] = float(((den01 - den01.mean()) ** 3).mean() / sd ** 3)
    f["q_kurt"] = float(((den01 - den01.mean()) ** 4).mean() / sd ** 4)
    return f


def spectral_features(img01: np.ndarray) -> dict:
    f = {}
    x = img01 - img01.mean()
    win = np.outer(np.hanning(x.shape[0]), np.hanning(x.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(x * win))
    P = np.abs(F) ** 2
    prof = _radial_profile(P, nbins=32)
    prof = prof / (prof.sum() + 1e-12)
    freqs = np.arange(32) + 0.5
    f["fft_centroid"] = float((prof * freqs).sum())
    cum = np.cumsum(prof)
    f["fft_f50"] = float(np.searchsorted(cum, 0.5))
    f["fft_f90"] = float(np.searchsorted(cum, 0.9))
    f["fft_low"] = float(prof[:4].sum())
    f["fft_mid"] = float(prof[4:12].sum())
    f["fft_high"] = float(prof[12:].sum())
    f["fft_peak_bin"] = float(np.argmax(prof[1:]) + 1)
    # angular anisotropy (direction invariant): power vs angle within an annulus
    h, w = P.shape
    cy, cx = h // 2, w // 2
    y, xg = np.indices(P.shape)
    r = np.hypot(y - cy, xg - cx)
    ang = np.arctan2(y - cy, xg - cx) % np.pi
    m = (r > 3) & (r < 40)
    f["fft_aniso"] = _circ_stats(ang[m], P[m])
    # autocorrelation length (Wiener-Khinchin)
    ac = np.real(np.fft.ifft2(np.fft.ifftshift(P)))
    ac = np.fft.fftshift(ac)
    ac = ac / (ac.max() + 1e-12)
    acp = _radial_profile(ac, nbins=64)
    rmax = min(cy, cx)
    rr = (np.arange(64) + 0.5) * rmax / 64

    def _len(th):
        below = np.where(acp < th)[0]
        return float(rr[below[0]]) if len(below) else float(rmax)

    f["ac_len_50"] = _len(0.5)
    f["ac_len_1e"] = _len(1 / np.e)
    f["ac_len_20"] = _len(0.2)
    return f


def structure_tensor_features(img01: np.ndarray) -> dict:
    f = {}
    for sigma in (4, 8):
        Axx, Axy, Ayy = feature.structure_tensor(img01, sigma=sigma, order="xy")
        tr = Axx + Ayy + 1e-9
        coh = np.sqrt((Axx - Ayy) ** 2 + 4 * Axy ** 2) / tr
        f[f"st_coh_mean_s{sigma}"] = float(coh.mean())
        f[f"st_coh_p75_s{sigma}"] = float(np.percentile(coh, 75))
        M = np.array([[Axx.mean(), Axy.mean()], [Axy.mean(), Ayy.mean()]])
        ev = np.linalg.eigvalsh(M)
        f[f"st_global_aniso_s{sigma}"] = float((ev[1] - ev[0]) / (ev[1] + ev[0] + 1e-12))
    return f


def glcm_features(den01: np.ndarray) -> dict:
    f = {}
    q = (den01 * 31).astype(np.uint8)
    dists = [1, 2, 4, 8, 16]
    g = graycomatrix(q, distances=dists, angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "homogeneity", "correlation", "energy"):
        v = graycoprops(g, prop)  # (dists, angles)
        for i, d in enumerate(dists):
            f[f"glcm_{prop}_d{d}"] = float(v[i].mean())
        f[f"glcm_{prop}_aniso"] = float(v[2].max() - v[2].min())
    return f


_GRAIN_KEYS = ("grain_area_mean", "grain_area_median", "grain_area_cv", "grain_area_logstd",
               "grain_area_p90", "grain_area_p10", "grain_diam_mean", "grain_ecc_mean",
               "grain_aspect_mean", "grain_aspect_p75", "grain_align", "grain_solidity",
               "grain_bnd_per_grain")


def grain_features(den01: np.ndarray) -> dict:
    """Segment grains from boundary edges; measure size, shape, alignment, uniformity."""
    f = {}
    edges, bnd, lab = _boundary_label(den01)
    f["edge_density"] = float((edges > 0).mean())
    props = [p for p in measure.regionprops(lab) if p.area >= 12]
    f["grain_count"] = float(len(props))
    if not props:
        f.update({k: 0.0 for k in _GRAIN_KEYS})
        return f
    areas = np.array([p.area for p in props], dtype=float)
    f["grain_area_mean"] = float(areas.mean())
    f["grain_area_median"] = float(np.median(areas))
    f["grain_area_cv"] = float(areas.std() / (areas.mean() + 1e-9))
    f["grain_area_logstd"] = float(np.log(areas).std())
    f["grain_area_p90"] = float(np.percentile(areas, 90))
    f["grain_area_p10"] = float(np.percentile(areas, 10))
    f["grain_diam_mean"] = float(np.mean(np.sqrt(areas / np.pi)) * 2)
    ecc = np.array([p.eccentricity for p in props])
    asp = np.array([p.axis_major_length / (p.axis_minor_length + 1e-6) for p in props])
    asp = np.clip(asp, 1, 10)
    ori = np.array([p.orientation for p in props]) % np.pi
    f["grain_ecc_mean"] = float(np.average(ecc, weights=areas))
    f["grain_aspect_mean"] = float(np.average(asp, weights=areas))
    f["grain_aspect_p75"] = float(np.percentile(asp, 75))
    f["grain_align"] = _circ_stats(ori, w=areas * np.clip(asp - 1, 0, 5) + 1e-6)
    f["grain_solidity"] = float(np.mean([p.solidity for p in props]))
    f["grain_bnd_per_grain"] = float(bnd.sum() / len(props))
    return f


def watershed_grain_features(den01: np.ndarray) -> dict:
    """Second, independent grain-size estimate via watershed on the gradient."""
    f = {}
    sm = filters.gaussian(den01, sigma=1.0)
    grad = filters.sobel(sm)
    for hmin in (0.01, 0.03):
        markers = measure.label(morphology.h_minima(grad, hmin))
        lab = segmentation.watershed(grad, markers)
        areas = np.bincount(lab.ravel())[1:]
        areas = areas[areas > 0].astype(float)
        tag = f"ws{int(hmin * 100)}"
        f[f"{tag}_count"] = float(len(areas))
        f[f"{tag}_area_mean"] = float(areas.mean()) if len(areas) else 0.0
        f[f"{tag}_area_cv"] = float(areas.std() / (areas.mean() + 1e-9)) if len(areas) else 0.0
    return f


def phase_features(den01: np.ndarray) -> dict:
    """Dark second phase (dark grains) and pores (small very-dark blobs)."""
    f = {}
    x = den01.ravel()
    rng = np.random.RandomState(0)
    s = x[rng.choice(len(x), 8000, replace=False)].reshape(-1, 1)
    for k in (2, 3):
        gm = GaussianMixture(k, random_state=0, n_init=2).fit(s)
        order = np.argsort(gm.means_.ravel())
        mu_ = gm.means_.ravel()[order]
        w_ = gm.weights_[order]
        sd_ = np.sqrt(gm.covariances_.ravel()[order])
        f[f"gmm{k}_dark_w"] = float(w_[0])
        f[f"gmm{k}_dark_mu"] = float(mu_[0])
        f[f"gmm{k}_sep"] = float((mu_[-1] - mu_[0]) / (sd_.mean() + 1e-6))
        if k == 3:
            f["gmm3_mid_w"] = float(w_[1])
    try:
        t = float(filters.threshold_otsu(den01))
    except ValueError:
        t = float(den01.mean())
    f["otsu_t"] = t
    f["frac_below_otsu"] = float((den01 < t).mean())
    mu, sd = float(den01.mean()), float(den01.std()) + 1e-6
    for k in (1.0, 1.5, 2.0):
        f[f"frac_dark_{k}"] = float((den01 < mu - k * sd).mean())
    # region-level second phase: grains whose mean intensity is clearly dark
    _, _, lab = _boundary_label(den01)
    props = [p for p in measure.regionprops(lab, intensity_image=den01) if p.area >= 12]
    if props:
        ints = np.array([p.intensity_mean for p in props])
        ar = np.array([p.area for p in props], dtype=float)
        thr = np.median(ints) - 0.6 * ints.std()
        dark = ints < thr
        f["phase2_area_frac"] = float(ar[dark].sum() / ar.sum())
        f["phase2_count_frac"] = float(dark.mean())
        f["grain_int_std"] = float(ints.std())
        f["grain_int_cv"] = float(ints.std() / (ints.mean() + 1e-9))
    else:
        f["phase2_area_frac"] = f["phase2_count_frac"] = f["grain_int_std"] = f["grain_int_cv"] = 0.0
    # pores: small, round, very dark blobs
    pore_mask = den01 < max(0.0, mu - 2.5 * sd)
    pore_mask = morphology.remove_small_objects(pore_mask, max_size=2)
    lab = measure.label(pore_mask)
    props = [p for p in measure.regionprops(lab) if 3 <= p.area <= 600]
    f["pore_count"] = float(len(props))
    f["pore_area_frac"] = float(sum(p.area for p in props) / den01.size)
    if props:
        f["pore_area_mean"] = float(np.mean([p.area for p in props]))
        f["pore_circ_mean"] = float(np.mean([4 * np.pi * p.area / (p.perimeter ** 2 + 1e-6) for p in props]))
        f["pore_solidity"] = float(np.mean([p.solidity for p in props]))
    else:
        f["pore_area_mean"] = f["pore_circ_mean"] = f["pore_solidity"] = 0.0
    f["frac_below_0p15"] = float((den01 < 0.15).mean())
    f["frac_below_0p25"] = float((den01 < 0.25).mean())
    return f


def lbp_features(den01: np.ndarray) -> dict:
    f = {}
    u8 = (den01 * 255).astype(np.uint8)
    for r in (1, 3):
        lbp = feature.local_binary_pattern(u8, 8, r, method="uniform")
        h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        for i, v in enumerate(h):
            f[f"lbp_r{r}_{i}"] = float(v)
    return f


# ----------------------------------------------------------------------------- entry point
def extract(img_u8: np.ndarray) -> dict:
    raw01 = img_u8.astype(np.float32) / 255.0
    den_u8 = cv2.fastNlMeansDenoising(img_u8, None, h=12, templateWindowSize=7, searchWindowSize=21)
    den01 = _stretch(den_u8.astype(np.float32) / 255.0)
    f = {}
    f.update(quality_features(raw01, den01))
    f.update(spectral_features(_stretch(raw01)))
    f.update({f"den_{k}": v for k, v in spectral_features(den01).items()})
    f.update(structure_tensor_features(den01))
    f.update(glcm_features(den01))
    f.update(grain_features(den01))
    f.update(watershed_grain_features(den01))
    f.update(phase_features(den01))
    f.update(lbp_features(den01))
    return f


# ----------------------------------------------------------------------------- v2 additions
def flat_region_grain_features(den01: np.ndarray) -> dict:
    """Grains = connected low-gradient (flat) regions. Voronoi grains have uniform
    intensity with thin dark boundaries, so this is more robust than Canny closure."""
    f = {}
    sm = filters.gaussian(den01, sigma=1.0)
    grad = filters.sobel(sm)
    for q in (35, 55):
        mask = grad < np.percentile(grad, q)
        mask = morphology.opening(mask, morphology.disk(1))
        lab = measure.label(mask, connectivity=1)
        props = [p for p in measure.regionprops(lab, intensity_image=den01) if p.area >= 8]
        tag = f"fr{q}"
        if not props:
            for k in ("count", "area_mean", "area_median", "area_logstd", "diam_mean", "aspect", "align", "dark_frac", "int_std"):
                f[f"{tag}_{k}"] = 0.0
            continue
        ar = np.array([p.area for p in props], dtype=float)
        f[f"{tag}_count"] = float(len(ar))
        f[f"{tag}_area_mean"] = float(ar.mean())
        f[f"{tag}_area_median"] = float(np.median(ar))
        f[f"{tag}_area_logstd"] = float(np.log(ar).std())
        f[f"{tag}_diam_mean"] = float(np.average(2 * np.sqrt(ar / np.pi), weights=ar))
        asp = np.clip(np.array([p.axis_major_length / (p.axis_minor_length + 1e-6) for p in props]), 1, 10)
        f[f"{tag}_aspect"] = float(np.average(asp, weights=ar))
        ori = np.array([p.orientation for p in props]) % np.pi
        f[f"{tag}_align"] = _circ_stats(ori, w=ar * np.clip(asp - 1, 0, 5) + 1e-6)
        ints = np.array([p.intensity_mean for p in props])
        thr = np.median(ints) - 0.6 * ints.std()
        f[f"{tag}_dark_frac"] = float(ar[ints < thr].sum() / ar.sum())
        f[f"{tag}_int_std"] = float(np.sqrt(np.average((ints - np.average(ints, weights=ar)) ** 2, weights=ar)))
    return f


def autocorr_shape_features(img01: np.ndarray) -> dict:
    """Second moments of the autocorrelation core -> scale-free elongation + size."""
    f = {}
    x = img01 - img01.mean()
    F = np.fft.fft2(x)
    ac = np.fft.fftshift(np.real(np.fft.ifft2(np.abs(F) ** 2)))
    ac = ac / (ac.max() + 1e-12)
    h, w = ac.shape
    cy, cx = h // 2, w // 2
    y, xg = np.indices(ac.shape)
    for th in (0.3, 0.5):
        m = ac > th
        # keep the connected component containing the centre
        lab = measure.label(m)
        m = lab == lab[cy, cx]
        wgt = np.clip(ac[m] - th, 0, None)
        dy, dx = (y[m] - cy).astype(float), (xg[m] - cx).astype(float)
        wsum = wgt.sum() + 1e-9
        Mxx, Myy, Mxy = (wgt * dx * dx).sum() / wsum, (wgt * dy * dy).sum() / wsum, (wgt * dx * dy).sum() / wsum
        ev = np.linalg.eigvalsh(np.array([[Mxx, Mxy], [Mxy, Myy]]))
        ev = np.clip(ev, 1e-9, None)
        f[f"ac{int(th*10)}_elong"] = float(np.sqrt(ev[1] / ev[0]))
        f[f"ac{int(th*10)}_size"] = float((ev[0] * ev[1]) ** 0.25)
        f[f"ac{int(th*10)}_area"] = float(m.sum())
    return f


def local_std_features(den01: np.ndarray) -> dict:
    f = {}
    for k in (5, 11):
        mu = cv2.blur(den01, (k, k))
        mu2 = cv2.blur(den01 * den01, (k, k))
        sd = np.sqrt(np.clip(mu2 - mu * mu, 0, None))
        f[f"lstd{k}_mean"] = float(sd.mean())
        f[f"lstd{k}_p25"] = float(np.percentile(sd, 25))
        f[f"lstd{k}_p90"] = float(np.percentile(sd, 90))
        f[f"lstd{k}_flat_frac"] = float((sd < 0.02).mean())
    return f


_extract_v1 = extract


def extract(img_u8: np.ndarray) -> dict:  # noqa: F811
    f = _extract_v1(img_u8)
    den_u8 = cv2.fastNlMeansDenoising(img_u8, None, h=12, templateWindowSize=7, searchWindowSize=21)
    den01 = _stretch(den_u8.astype(np.float32) / 255.0)
    f.update(flat_region_grain_features(den01))
    f.update(autocorr_shape_features(den01))
    f.update(local_std_features(den01))
    return f
