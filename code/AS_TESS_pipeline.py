import os
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from astropy.timeseries import LombScargle
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.signal import find_peaks
import pandas as pd
from scipy.linalg import lstsq
from astroquery.gaia import Gaia
import re
import json
from decimal import Decimal, InvalidOperation
from tqdm.auto import tqdm

_CORRUPT_FITS_RE = re.compile(r"supported data product:\s*(.*\.fits)", re.IGNORECASE)

# -----------------------------
# Global LS frequency defaults (fixed grid)
# -----------------------------
DEFAULT_FMIN = 0.004     # 1/day  (Pmax = 250 d)
DEFAULT_FMAX = 5.0       # 1/day  (Pmin = 0.2 d)
N_FREQ_DEFAULT = 240000  # fixed number of frequency samples for ALL stars


# -----------------------------
# Helpers
# -----------------------------
def _to_float_or_nan(x):
    try:
        if x is None:
            return np.nan
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return np.nan
        return float(s)
    except Exception:
        return np.nan


def build_fig_suptitle(label, per=None, psini=None, gaia_period=None, gmag=None):
    """
    Returns a consistent suptitle string.

    New-table behavior:
      "<label> | Gao P_rot (Per) = ... d | Gaia P/sini (Psini) = ... d | M_G = ..."

    Backwards-compatible fallback (if Per/Psini are missing):
      "<label> | Pasquale's period = ... d | M_G = ..."
    """
    g = _to_float_or_nan(gmag)
    p_per = _to_float_or_nan(per)
    p_psini = _to_float_or_nan(psini)
    p_legacy = _to_float_or_nan(gaia_period)

    parts = [str(label).strip() if label is not None else ""]

    # Prefer new table columns if present
    if np.isfinite(p_per) or np.isfinite(p_psini):
        if np.isfinite(p_per):
            parts.append(f"Gao P_rot (Per) = {p_per:.6f} d")
        else:
            parts.append("Gao P_rot (Per) = NA")

        if np.isfinite(p_psini):
            parts.append(f"Gaia P/sini (Psini) = {p_psini:.6f} d")
        else:
            parts.append("Gaia P/sini (Psini) = NA")

    else:
        # Legacy fallback
        if np.isfinite(p_legacy):
            parts.append(f"Pasquale's period = {p_legacy:.6f} d")
        else:
            parts.append("Pasquale's period = NA")

    if np.isfinite(g):
        parts.append(f"M_G = {g:.3f}")
    else:
        parts.append("G = NA")

    return " | ".join(parts)

def smart_format(val):
    import numpy as np
    if np.isnan(val):
        return "nan"
    if np.abs(val) < 1e-8:
        return "<1e-8"
    if np.abs(val) < 1e-3 and val != 0:
        return f"{val:.1e}"
    else:
        return f"{val:.5f}"

def make_design_matrix_and_weighted_mag_of_N_harmonics(t, mag_error, mag, N):
    M = np.zeros((len(t), 2 * N + 1))
    pi = np.pi

    M[:, 0] = 1 / mag_error  # constant term
    for n in range(1, N + 1):
        M[:, 2*n - 1] = np.cos(2 * pi * n * t) / mag_error
        M[:, 2*n]     = np.sin(2 * pi * n * t) / mag_error

    wmag = mag / mag_error
    return M, wmag

def Make_Unweighted_Matrix (t, N=3):
    M = np.zeros((len(t), 2 * N + 1))

    M[:, 0] = 1  # constant term
    for n in range(1, N + 1):
        M[:, 2 * n - 1] = np.cos(2 * np.pi * n * t)
        M[:, 2 * n] = np.sin(2 * np.pi * n * t)
    return M


def GetErrors_Of_Fitted_Paramters (matrix, par, mag, t, err, N=3) :

    unweighted_mat = Make_Unweighted_Matrix(t,N)

    residuals = (mag - unweighted_mat.dot(par)) / err

    # Calculate the variance-covariance matrix
    MSE = np.sum(residuals ** 2) / (len(mag) - len(par))
    covariance_matrix = MSE * np.linalg.inv(matrix.T.dot(matrix))

    # Calculate the standard errors
    par_error = np.sqrt(np.diag(covariance_matrix))
    return par_error

def Y_values_of_fitted_function_N(fit_par, x, N):
    y = fit_par[0]
    for n in range(1, N + 1):
        y += fit_par[2*n - 1] * np.cos(2 * np.pi * n * x)
        y += fit_par[2*n]     * np.sin(2 * np.pi * n * x)
    return y


def fitfunction_all_points(mag, error, folded_time, N=3):
    matrix, weighted_mag = make_design_matrix_and_weighted_mag_of_N_harmonics(folded_time, error, mag, N)
    fit_par, res, rnk, s = lstsq(matrix, weighted_mag)
    fit_par_error = GetErrors_Of_Fitted_Paramters(matrix, fit_par, mag, folded_time, error, N)
    x = np.linspace(0, 1, 100)

    y = Y_values_of_fitted_function_N(fit_par, x, N)
    return fit_par, fit_par_error, y


def make_fit_quadrature_string_N(par, par_err, N):
    """
    Print quadrature values and their errors for N harmonics.
    """
    fit_par_str = f"a0 = {smart_format(par[0])} ± {smart_format(par_err[0])}\n"
    for i in range(1, N + 1):
        a_cos = par[2 * i - 1]
        a_sin = par[2 * i]
        a_cos_err = par_err[2 * i - 1]
        a_sin_err = par_err[2 * i]

        amplitude = np.sqrt(a_cos ** 2 + a_sin ** 2)
        # error propagation for amplitude
        amplitude_err = np.sqrt(
            (a_cos * a_cos_err / amplitude) ** 2 +
            (a_sin * a_sin_err / amplitude) ** 2
        ) if amplitude != 0 else 0

        fit_par_str += (
            f"a{i} = {smart_format(amplitude)} ± {smart_format(amplitude_err)}\n"
        )
    return fit_par_str




def sanitize_target_id(tic_id=None, gaia_source_id=None, ra=None, dec=None):
    """Filesystem-safe identifier for output folders."""
    if tic_id is not None and str(tic_id).strip() != "":
        return f"TIC_{str(tic_id).strip()}"
    if gaia_source_id is not None and str(gaia_source_id).strip() != "":
        return f"GAIA_{str(gaia_source_id).strip()}"
    return f"RA_{float(ra):.6f}_DEC_{float(dec):.6f}"
def estimate_flat_floor_and_nonflat_max(freq, power, hi_frac=0.20, smooth_frac=0.02, k=3.0):
    """
    Estimate the high-frequency 'flat' noise floor and find the highest frequency
    where the *smoothed* log-power rises significantly above that floor.

    Returns:
      floor_power (float)      : estimated flat floor in linear power units
      floor_lp (float)         : log10(floor_power)
      floor_mad_lp (float)     : MAD in log10(power) in the high-freq region
      f_nonflat_max (float)    : max frequency of non-flat region (based on smoothed log-power)
      lp_smooth (np.ndarray)   : smoothed log10(power) curve
      thr_lp (float)           : threshold in log10(power) defining "non-flat"
    """
    f = np.asarray(freq, float)
    p = np.asarray(power, float)

    good = np.isfinite(f) & np.isfinite(p)
    if np.sum(good) < 100:
        return np.nan, np.nan, np.nan, np.nan, None, np.nan

    # avoid log(0) / log(neg): clip to small positive epsilon
    p_good = p[good]
    med_p = np.nanmedian(p_good)
    eps = (med_p * 1e-6 + 1e-12) if np.isfinite(med_p) else 1e-12
    p_clip = np.clip(p, eps, None)

    lp = np.log10(p_clip)
    lp_med = np.nanmedian(lp[good])
    lp = np.where(np.isfinite(lp), lp, lp_med)

    n = len(lp)
    # smoothing window as a fraction of array length
    w = int(max(101, round(smooth_frac * n)))
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=float) / w
    lp_smooth = np.convolve(lp, kernel, mode="same")

    # define high-frequency region as the top hi_frac of frequencies
    i0 = int(np.floor((1.0 - hi_frac) * n))
    hi = lp_smooth[i0:]

    floor_lp = float(np.nanmedian(hi))
    floor_mad_lp = float(1.4826 * np.nanmedian(np.abs(hi - floor_lp)))
    if (not np.isfinite(floor_mad_lp)) or floor_mad_lp == 0:
        floor_mad_lp = float(np.nanstd(hi))

    thr_lp = floor_lp + k * floor_mad_lp

    # non-flat region: where smoothed log-power exceeds the threshold
    idx = np.where(lp_smooth > thr_lp)[0]
    if idx.size == 0:
        f_nonflat_max = float(f.max())
    else:
        f_nonflat_max = float(f[int(idx.max())])

    floor_power = float(10 ** floor_lp)
    return floor_power, floor_lp, floor_mad_lp, f_nonflat_max, lp_smooth, float(thr_lp)

def pick_flux_columns(lc, author="", flux_mode="default"):
    """
    Decide which flux column we will use.

    flux_mode:
      - "detrended": SPOC->PDCSAP; QLP->DET_FLUX/KSPSAP; fallback->SAP
      - "raw":       SAP (if present), else fallback to detrended choices
      - "default":   SPOC->PDCSAP; QLP->SAP  (per your requested defaults)
    Returns:
      flux_kind (str), flux_col (str|None), flux_err_col (str|None)
    """
    try:
        cols_map = {c.lower(): c for c in lc.colnames}  # preserves real case
    except Exception:
        cols_map = {}

    auth = (author or "").strip().lower()
    mode = (flux_mode or "default").strip().lower()

    def has(col_lower):
        return col_lower in cols_map

    def get(col_lower):
        return cols_map.get(col_lower, None)

    def ret(kind, f_lower, e_lower):
        fc = get(f_lower)
        ec = get(e_lower) if e_lower else None
        return kind, fc, ec

    # -------------------------
    # RAW request: prefer SAP everywhere
    # -------------------------
    if mode == "raw":
        if has("sap_flux"):
            return ret("SAP", "sap_flux", "sap_flux_err")
        # If SAP doesn't exist, fall back to detrended logic
        mode = "detrended"

    # -------------------------
    # DEFAULT request: dont change the given SPOC->PDCSAP ; QLP->SAP
    # -------------------------
    if mode == "default":
        if "spoc" in auth:
            if "pdcsap_flux" in lc.meta["FLUX_ORIGIN"]:
                return ret("PDCSAP", None, None)
            if "sap_flux" in lc.meta["FLUX_ORIGIN"]:
                return ret("SAP", None, None)
            return "UNKNOWN", None, None

        if "qlp" in auth:
            # your requested default for QLP is SAP (raw)
            if "sap_flux" in lc.meta["FLUX_ORIGIN"]:
                return ret("SAP", None, None)
            # fallback if SAP isn't there
            if "det_flux" in lc.meta["FLUX_ORIGIN"]:
                return ret("DET_FLUX", None, None)
            if "kspsap_flux" in lc.meta["FLUX_ORIGIN"]:
                return ret("KSPSAP_FLUX", None, None)
            return "UNKNOWN", None, None

        # author unknown: reasonable "default" guess
        if "pdcsap_flux" in lc.meta["FLUX_ORIGIN"]:
            return ret("PDCSAP", None, None)
        if "sap_flux" in lc.meta["FLUX_ORIGIN"]:
            return ret("SAP", None, None)
        if "det_flux" in lc.meta["FLUX_ORIGIN"]:
            return ret("DET_FLUX", None, None)
        if "kspsap_flux" in lc.meta["FLUX_ORIGIN"]:
            return ret("KSPSAP_FLUX", None, None)
        return "UNKNOWN", None, None

    # -------------------------
    # DETRENDED request
    # -------------------------
    if mode == "detrended":
        if "spoc" in auth:
            if has("pdcsap_flux"):
                return ret("PDCSAP", "pdcsap_flux", "pdcsap_flux_err")
            if has("sap_flux"):
                return ret("SAP", "sap_flux", "sap_flux_err")
            return "UNKNOWN", None, None

        if "qlp" in auth:
            if has("det_flux"):
                return ret("DET_FLUX", "det_flux", "det_flux_err")
            if has("kspsap_flux"):
                return ret("KSPSAP_FLUX", "kspsap_flux", "kspsap_flux_err")
            if has("sap_flux"):
                return ret("SAP", "sap_flux", "sap_flux_err")
            return "UNKNOWN", None, None

        # author unknown: prefer detrended-ish columns
        if has("pdcsap_flux"):
            return ret("PDCSAP", "pdcsap_flux", "pdcsap_flux_err")
        if has("det_flux"):
            return ret("DET_FLUX", "det_flux", "det_flux_err")
        if has("kspsap_flux"):
            return ret("KSPSAP_FLUX", "kspsap_flux", "kspsap_flux_err")
        if has("sap_flux"):
            return ret("SAP", "sap_flux", "sap_flux_err")

    return "UNKNOWN", None, None


def local_periodogram_floor(power, best_idx, half_width=3000, exclude=200):
    """
    Local baseline around best peak: median power in a window around best_idx,
    excluding a small region near the peak to avoid contaminating the floor.

    Returns: local_floor (float)
    """
    p = np.asarray(power, float)
    n = len(p)
    lo = max(0, int(best_idx) - int(half_width))
    hi = min(n, int(best_idx) + int(half_width) + 1)

    seg = p[lo:hi].astype(float, copy=True)
    c = int(best_idx) - lo

    ex_lo = max(0, c - int(exclude))
    ex_hi = min(len(seg), c + int(exclude) + 1)
    seg[ex_lo:ex_hi] = np.nan

    floor = float(np.nanmedian(seg))
    return floor if np.isfinite(floor) and floor > 0 else np.nan

def parse_int_like(x, field_name="id"):
    """
    Robust parse for integer-like IDs that may appear as:
      - '5353078432665113'
      - '5353078432665113.0'
      - '5.353078432665113e+18'
    Returns: (value_int, warning_str_or_empty)
    """
    if x is None:
        raise ValueError(f"{field_name} is None")

    s = str(x).strip()
    if s == "" or s.lower() == "nan":
        raise ValueError(f"{field_name} is empty")

    warning = ""
    # Heuristic: scientific notation often means the value may have been rounded in the CSV
    if ("e" in s.lower()):
        warning = f"{field_name}_sci_notation_possible_precision_loss"

    try:
        # Decimal handles scientific notation safely (as text)
        val = int(Decimal(s))
        return val, warning
    except (InvalidOperation, ValueError):
        # fallback: strip trailing .0
        try:
            val = int(s.split(".")[0])
            return val, warning or f"{field_name}_fallback_splitdot"
        except Exception as e:
            raise ValueError(f"Could not parse {field_name}='{s}': {e}")


_GAIA_COORD_CACHE = {}  # source_id(int) -> (ra_deg, dec_deg)

def safe_download(sr, max_tries=2):
    for _ in range(max_tries):
        try:
            return sr.download()
        except Exception as e:
            msg = str(e)
            m = _CORRUPT_FITS_RE.search(msg)
            if m:
                bad_path = m.group(1).strip()
                if os.path.exists(bad_path):
                    try:
                        os.remove(bad_path)
                        print(f"Deleted corrupt cached file: {bad_path}")
                        continue
                    except Exception:
                        pass
            raise
    return None

def gaia_source_id_to_radec(source_id):
    sid, warn = parse_int_like(source_id, field_name="gaia_source_id")

    if sid in _GAIA_COORD_CACHE:
        return _GAIA_COORD_CACHE[sid]

    query = f"""
        SELECT ra, dec
        FROM gaiadr3.gaia_source
        WHERE source_id = {sid}
    """
    job = Gaia.launch_job_async(query)
    res = job.get_results()

    if len(res) == 0:
        raise ValueError(f"Gaia DR3 source_id not found: {sid}")

    ra = float(res["ra"][0])
    dec = float(res["dec"][0])

    _GAIA_COORD_CACHE[sid] = (ra, dec)
    return ra, dec

def resolve_plot_paths(output_dir, target_id, layout="by_star"):
    """
    layout:
      - "by_star"      : output_dir/<target_id>/periodogram_and_folded.png, per_sector_raw_and_folded.png
      - "by_plot_type" : output_dir/periodogram_and_folded/<target_id>.png
                         output_dir/per_sector_raw_and_folded/<target_id>.png

    Always returns (star_dir, path_periodogram_png, path_per_sector_png).
    star_dir is still created so your cleaned CSV can live somewhere consistent.
    """
    layout = (layout or "by_star").strip().lower()

    star_dir = os.path.join(output_dir, target_id)
    os.makedirs(star_dir, exist_ok=True)

    if layout in {"by_plot_type", "plot_type", "type"}:
        pg_dir = os.path.join(output_dir, "periodogram_and_folded")
        ps_dir = os.path.join(output_dir, "per_sector_raw_and_folded")
        os.makedirs(pg_dir, exist_ok=True)
        os.makedirs(ps_dir, exist_ok=True)

        pg_path = os.path.join(pg_dir, f"{target_id}.png")
        ps_path = os.path.join(ps_dir, f"{target_id}.png")
        return star_dir, pg_path, ps_path

    # default: by star folder
    pg_path = os.path.join(star_dir, "periodogram_and_folded.png")
    ps_path = os.path.join(star_dir, "per_sector_raw_and_folded.png")
    return star_dir, pg_path, ps_path

def _cached_file_paths(star_dir):
    return {
        "cleaned_lc": os.path.join(star_dir, "cleaned_lightcurve.csv"),
        "pg_npz": os.path.join(star_dir, "periodogram.npz"),
        "pg_meta": os.path.join(star_dir, "periodogram_meta.json"),
        "sector_pg_npz": os.path.join(star_dir, "sector_periodograms.npz"),
    }

def _has_core_cache(star_dir):
    p = _cached_file_paths(star_dir)
    return (os.path.exists(p["cleaned_lc"]) and os.path.exists(p["pg_npz"]) and os.path.exists(p["pg_meta"]))

def _mode_string(values, default="unknown"):
    """Most-common string in an array/Series."""
    try:
        s = pd.Series(values).astype(str)
        s = s[(s != "") & (s.str.lower() != "nan")]
        if len(s) == 0:
            return default
        return str(s.value_counts().idxmax())
    except Exception:
        return default

def fill_metrics_from_cache(metrics, star_dir, per_sector_periodograms=False):
    """
    Populate *all* metrics fields using cached products in star_dir:
      - cleaned_lightcurve.csv
      - periodogram.npz
      - periodogram_meta.json
      - (optional) sector_periodograms.npz

    No downloads. No LS recomputation (unless you explicitly choose to do per-sector LS
    and the sector cache is missing).
    """
    freq, power, pg_meta, lc_df, sector_cache = _load_cached_products(star_dir)

    # --- paths ---
    metrics["outdir"] = star_dir
    metrics["cleaned_lightcurve_csv"] = os.path.join(star_dir, "cleaned_lightcurve.csv")
    metrics["periodogram_npz"] = os.path.join(star_dir, "periodogram.npz")
    metrics["periodogram_meta_json"] = os.path.join(star_dir, "periodogram_meta.json")
    metrics["periodogram_csv_gz"] = os.path.join(star_dir, "periodogram.csv.gz")
    metrics["sector_periodograms_npz"] = os.path.join(star_dir, "sector_periodograms.npz")

    # --- lightcurve stats (use GOOD points exactly like your LS run) ---
    required = {"sector", "author", "flux_kind", "time_btjd", "flux_norm", "flux_err", "quality"}
    missing = required - set(lc_df.columns)
    if missing:
        raise ValueError(f"Cached cleaned_lightcurve.csv missing columns: {sorted(missing)}")

    sector_all = np.asarray(lc_df["sector"], int)
    time_all   = np.asarray(lc_df["time_btjd"], float)
    flux_all   = np.asarray(lc_df["flux_norm"], float)
    ferr_all   = np.asarray(lc_df["flux_err"], float)
    qual_all   = np.asarray(lc_df["quality"], np.int64)

    good_all = (qual_all == 0) & np.isfinite(time_all) & np.isfinite(flux_all) & np.isfinite(ferr_all)

    metrics["n_points_total_all"] = int(len(time_all))
    metrics["n_points_good_total"] = int(np.sum(good_all))
    metrics["n_points_flagged_total"] = int(np.sum(~good_all))

    if np.sum(good_all) == 0:
        raise ValueError("Cached cleaned lightcurve has zero good points (quality==0).")

    time = time_all[good_all]
    flux = flux_all[good_all]

    metrics["time_min_btjd"] = float(np.nanmin(time))
    metrics["time_max_btjd"] = float(np.nanmax(time))
    metrics["baseline_days"] = float(np.nanmax(time) - np.nanmin(time))
    metrics["n_points_total"] = int(len(time))
    metrics["flux_mean"] = float(np.nanmean(flux))
    metrics["flux_std"]  = float(np.nanstd(flux))

    # --- per-sector counts etc. (reconstructed from cleaned LC) ---
    uniq_secs = np.unique(sector_all)
    metrics["n_downloaded_sectors"] = int(len(uniq_secs))
    metrics["n_usable_sectors"] = int(len(uniq_secs))
    metrics["sector_numbers_downloaded"] = json.dumps([int(s) for s in uniq_secs.tolist()])

    # authors / flux kinds (from cleaned LC)
    try:
        metrics["authors_used"] = json.dumps(sorted(set(lc_df["author"].astype(str).tolist())))
    except Exception:
        metrics["authors_used"] = ""
    try:
        metrics["flux_kinds_used"] = json.dumps(sorted(set(lc_df["flux_kind"].astype(str).tolist())))
    except Exception:
        metrics["flux_kinds_used"] = ""

    # counts per sector (post-filter, since that’s all that exists in cache)
    raw_pts = []
    good_pts = []
    flag_pts = []
    kept_frac = []

    qb_present = []
    qn_present = []
    qs_soft = []

    for s in uniq_secs:
        m = (sector_all == s)
        n_tot = int(np.sum(m))
        n_good = int(np.sum(m & (qual_all == 0)))
        n_flag = int(np.sum(m & (qual_all != 0)))

        raw_pts.append(n_tot)
        good_pts.append(n_good)
        flag_pts.append(n_flag)

        # "kept frac" can’t be reconstructed to mean the original outlier-filtering fraction.
        # We fill it with the *good fraction* (useful + consistent), so the table is fully populated.
        kept_frac.append(float(n_good / n_tot) if n_tot > 0 else np.nan)

        bits, names = quality_present_lists(qual_all[m])
        qb_present.append(bits)
        qn_present.append(names)
        qs_soft.append(
            quality_summary(
                qual_all[m], frac_thresh=0.30, max_flags=2, denom="all",
                ignore_mask=HARD_QUALITY_MASK, prefix="Soft"
            )
        )

    metrics["raw_points_per_sector"] = json.dumps(raw_pts)
    metrics["kept_points_per_sector"] = json.dumps(raw_pts)          # post-filter counts
    metrics["kept_frac_per_sector"] = json.dumps(kept_frac)          # good fraction proxy

    metrics["good_points_per_sector"] = json.dumps(good_pts)
    metrics["flagged_points_per_sector"] = json.dumps(flag_pts)
    metrics["quality_bits_present_per_sector"] = json.dumps(qb_present)
    metrics["quality_names_present_per_sector"] = json.dumps(qn_present)
    metrics["quality_soft_summary_per_sector"] = json.dumps(qs_soft)

    # --- periodogram stats (from cached freq/power) ---
    freq = np.asarray(freq, float)
    power = np.asarray(power, float)

    metrics["n_freq"] = int(len(freq))
    metrics["fmin"] = float(np.nanmin(freq))
    metrics["fmax"] = float(np.nanmax(freq))

    metrics["power_min"] = float(np.nanmin(power))
    metrics["power_max"] = float(np.nanmax(power))
    metrics["power_median"] = float(np.nanmedian(power))
    metrics["threshold"] = float(metrics["power_median"] * 1.2) if np.isfinite(metrics["power_median"]) else np.nan

    thr = metrics["threshold"]
    peaks, _ = find_peaks(power, height=thr if np.isfinite(thr) else None)
    metrics["n_peaks"] = int(len(peaks))

    if len(peaks) == 0:
        best_idx = int(np.nanargmax(power))
        metrics["note"] = "(fallback: max power)"
    else:
        best_idx = int(peaks[np.nanargmax(power[peaks])])
        metrics["note"] = "(best significant peak)"

    best_freq = float(freq[best_idx])
    best_period = float(1.0 / best_freq) if best_freq > 0 else np.nan

    # reject P > baseline (same logic)
    T = float(metrics["baseline_days"])
    if np.isfinite(best_period) and np.isfinite(T) and (best_period > T) and (T > 0):
        sorted_idx = np.argsort(power)[::-1]
        for idx in sorted_idx:
            cand = 1.0 / float(freq[idx])
            if cand <= T:
                best_idx = int(idx)
                best_freq = float(freq[best_idx])
                best_period = float(cand)
                metrics["note"] = "(corrected: within baseline)"
                break

    metrics["best_idx"] = int(best_idx)
    metrics["best_freq"] = float(best_freq)
    metrics["best_period_days"] = float(best_period)
    metrics["best_power"] = float(power[best_idx])

    # ratios
    med = float(metrics["power_median"])
    metrics["best_power_over_median"] = float(metrics["best_power"] / med) if (np.isfinite(med) and med > 0) else np.nan

    # floors / non-flat diagnostics
    floor_power, floor_lp, floor_mad_lp, f_nonflat_max, lp_smooth, thr_lp = \
        estimate_flat_floor_and_nonflat_max(freq, power, hi_frac=0.20, smooth_frac=0.02, k=3.0)

    metrics["power_floor_flat"] = float(floor_power) if np.isfinite(floor_power) else np.nan
    metrics["floor_log10"] = float(floor_lp) if np.isfinite(floor_lp) else np.nan
    metrics["floor_mad_log10"] = float(floor_mad_lp) if np.isfinite(floor_mad_lp) else np.nan
    metrics["f_nonflat_max"] = float(f_nonflat_max) if np.isfinite(f_nonflat_max) else np.nan

    if np.isfinite(floor_power) and floor_power > 0:
        metrics["best_power_over_flat_floor"] = float(metrics["best_power"] / floor_power)
    else:
        metrics["best_power_over_flat_floor"] = np.nan

    local_floor = local_periodogram_floor(power, best_idx, half_width=3000, exclude=200)
    metrics["power_floor_local"] = float(local_floor) if np.isfinite(local_floor) else np.nan
    if np.isfinite(local_floor) and local_floor > 0:
        metrics["best_power_over_local_floor"] = float(metrics["best_power"] / local_floor)
    else:
        metrics["best_power_over_local_floor"] = np.nan

    # FWHM period error
    half_max = float(metrics["best_power"]) / 2.0
    left_idxs = np.where(power[:best_idx] < half_max)[0]
    left_idx = int(left_idxs[-1]) if left_idxs.size > 0 else 0
    right_idxs = np.where(power[best_idx:] < half_max)[0]
    right_idx = int(best_idx + right_idxs[0]) if right_idxs.size > 0 else (len(power) - 1)

    fwhm = float(freq[right_idx] - freq[left_idx])
    metrics["period_err_days"] = float(fwhm / (best_freq ** 2)) if (np.isfinite(best_freq) and best_freq > 0) else np.nan

    # T0 (matches your definition)
    metrics["T0_btjd"] = float(np.floor(np.nanmin(time)))

    # --- per-sector LS summary (only if requested and we have cached sector powers) ---
    if per_sector_periodograms and (sector_cache is not None) and ("power" in sector_cache) and ("sectors" in sector_cache):
        sec_ids = np.asarray(sector_cache["sectors"], int)
        pow2d = np.asarray(sector_cache["power"], float)
        best_per_s = np.asarray(sector_cache["best_period"], float)
        perr_s = np.asarray(sector_cache["period_err"], float)
        peak_over_med_s = np.asarray(sector_cache["peak_over_median"], float)

        sector_ls = []
        for i, sec in enumerate(sec_ids):
            pbest = float(best_per_s[i])
            perr  = float(perr_s[i])
            pwr = pow2d[i]
            # compute local floor ratio per-sector (not stored)
            bidx = int(np.nanargmax(pwr)) if np.any(np.isfinite(pwr)) else 0
            loc = local_periodogram_floor(pwr, bidx, half_width=3000, exclude=200)
            bpow = float(pwr[bidx]) if np.isfinite(pwr[bidx]) else np.nan
            rloc = float(bpow / loc) if (np.isfinite(bpow) and np.isfinite(loc) and loc > 0) else np.nan

            sector_ls.append({
                "sector": int(sec),
                "author": "cached",
                "flux_kind": "cached",
                "best_period_days": pbest,
                "period_err_days": perr,
                "best_power": bpow,
                "best_power_over_median": float(peak_over_med_s[i]) if np.isfinite(peak_over_med_s[i]) else np.nan,
                "best_power_over_local_floor": rloc,
            })

        metrics["sector_ls_json"] = json.dumps(sector_ls)

    return metrics

def _load_cached_products(star_dir):
    p = _cached_file_paths(star_dir)

    lc_df = pd.read_csv(p["cleaned_lc"])
    with np.load(p["pg_npz"]) as z:
        freq = np.asarray(z["freq"], float)
        power = np.asarray(z["power"], float)

    with open(p["pg_meta"], "r") as f:
        pg_meta = json.load(f)

    sector_cache = None
    if os.path.exists(p["sector_pg_npz"]):
        with np.load(p["sector_pg_npz"]) as z:
            # All arrays are numeric; no allow_pickle needed
            sector_cache = {k: z[k] for k in z.files}

    return freq, power, pg_meta, lc_df, sector_cache


def recreate_figures_from_cached(
    *,
    star_dir,
    fig1_path,
    fig2_path,
    fig_title,
    interactive,
    plot_by_sector_default,
    per_sector_periodograms,
    show_flagged_points=False,
):
    """
    Rebuild figures using:
      - cleaned_lightcurve.csv
      - periodogram.npz + periodogram_meta.json
      - (optional) sector_periodograms.npz for per-sector LS panels
    """
    freq, power, pg_meta, lc_df, sector_cache = _load_cached_products(star_dir)

    # Decide whether we can do per-sector LS panels from cache
    have_sector_ls_cache = (sector_cache is not None) and ("power" in sector_cache) and ("sectors" in sector_cache)
    use_sector_ls_panels = bool(per_sector_periodograms and have_sector_ls_cache)

    if per_sector_periodograms and not use_sector_ls_panels:
        print("Cached run found, but no saved sector_periodograms.npz -> recreating per-sector figure WITHOUT LS panels.")

    # Plot-by-sector choice (same behavior as main)
    if interactive:
        sector_choice = input("Plot each sector individually? [y/N]: ").strip().lower()
        plot_by_sector = sector_choice in ["y", "yes"]
    else:
        plot_by_sector = bool(plot_by_sector_default)

    # Pull arrays from cleaned LC
    required = {"sector", "author", "flux_kind", "time_btjd", "flux_norm", "flux_err", "quality"}
    missing = required - set(lc_df.columns)
    if missing:
        raise ValueError(f"Cached cleaned_lightcurve.csv missing columns: {sorted(missing)}")

    sector_all = np.asarray(lc_df["sector"], int)
    time_all = np.asarray(lc_df["time_btjd"], float)
    flux_all = np.asarray(lc_df["flux_norm"], float)
    ferr_all = np.asarray(lc_df["flux_err"], float)
    qual_all = np.asarray(lc_df["quality"], np.int64)

    good_all = (qual_all == 0) & np.isfinite(time_all) & np.isfinite(flux_all) & np.isfinite(ferr_all)
    time = time_all[good_all]
    flux = flux_all[good_all]
    flux_err = ferr_all[good_all]

    if time.size == 0:
        raise ValueError("Cached cleaned lightcurve has zero good points (quality==0).")

    if np.all(flux_err == 0):
        flux_err = None

    # best period info from meta (fallback if absent)
    best_freq = float(pg_meta.get("best_freq", np.nan))
    best_period = float(pg_meta.get("best_period_days", np.nan))
    period_err = float(pg_meta.get("period_err_days", np.nan))
    threshold = float(pg_meta.get("threshold", np.nan))

    if not np.isfinite(best_freq) or not np.isfinite(best_period):
        # fallback: recompute best from cached power only (NO re-download / no LS recomputation)
        best_idx = int(np.nanargmax(power))
        best_freq = float(freq[best_idx])
        best_period = float(1.0 / best_freq) if best_freq > 0 else np.nan
        # crude FWHM error from cached curve only
        half_max = float(power[best_idx]) / 2.0
        left_idxs = np.where(power[:best_idx] < half_max)[0]
        left_idx = int(left_idxs[-1]) if left_idxs.size > 0 else 0
        right_idxs = np.where(power[best_idx:] < half_max)[0]
        right_idx = int(best_idx + right_idxs[0]) if right_idxs.size > 0 else (len(power) - 1)
        fwhm = float(freq[right_idx] - freq[left_idx])
        period_err = float(fwhm / (best_freq ** 2)) if best_freq > 0 else np.nan
        if not np.isfinite(threshold):
            threshold = float(np.nanmedian(power) * 1.2)

    # get a stable best_idx for annotations/titles
    if np.isfinite(best_freq):
        best_idx = int(np.nanargmin(np.abs(freq - best_freq)))
    else:
        best_idx = int(np.nanargmax(power))

    pmed = float(np.nanmedian(power))
    peak_over_median = float(power[best_idx] / pmed) if (np.isfinite(pmed) and pmed > 0) else np.nan


    T0 = float(np.floor(np.nanmin(time)))
    T = float(np.nanmax(time) - np.nanmin(time))

    # Global y-limits like your code (based on GOOD points)
    low_global, high_global = np.nanpercentile(flux, [0.1, 99])
    ymin_global = low_global - 0.1 * (high_global - low_global)
    ymax_global = high_global + 0.1 * (high_global - low_global)

    # Build per-sector lists from cached LC
    times, fluxes, fluxerrs, quals, meta = [], [], [], [], []
    for sec in np.unique(sector_all):
        m = (sector_all == sec)
        t_sec = time_all[m]
        f_sec = flux_all[m]
        e_sec = ferr_all[m]
        q_sec = qual_all[m]

        # author/flux_kind per sector from mode
        a_sec = _mode_string(lc_df.loc[m, "author"].values, default="unknown")
        fk_sec = _mode_string(lc_df.loc[m, "flux_kind"].values, default="UNKNOWN")

        # minimal meta reconstruction
        mi = {
            "sector": int(sec),
            "author": a_sec,
            "flux_kind": fk_sec,
            "n_raw": int(len(t_sec)),
            "n_outliers_removed": 0,  # unknown from cache (you did not store pre-filter n_raw)
            "n_kept_good": int(np.sum(q_sec == 0)),
            "n_kept_flagged": int(np.sum(q_sec != 0)),
            "quality_bits_present_raw": quality_present_lists(q_sec)[0],
            "quality_names_present_raw": quality_present_lists(q_sec)[1],
            "quality_soft_summary_raw": quality_summary(
                q_sec, frac_thresh=0.30, max_flags=2, denom="all",
                ignore_mask=HARD_QUALITY_MASK, prefix="Soft"
            )
        }

        times.append(np.asarray(t_sec, float))
        fluxes.append(np.asarray(f_sec, float))
        fluxerrs.append(np.asarray(e_sec, float))
        quals.append(np.asarray(q_sec, np.int64))
        meta.append(mi)

    # sort by sector (keep -1 first if present)
    order = np.argsort([m["sector"] for m in meta])
    times = [times[i] for i in order]
    fluxes = [fluxes[i] for i in order]
    fluxerrs = [fluxerrs[i] for i in order]
    quals = [quals[i] for i in order]
    meta = [meta[i] for i in order]
    n_sectors = len(times)

    # ---------- plotting helpers (same as your code) ----------
    def _plot_fold(ax, t_sec, f_sec, e_sec, q_sec, period, T0, title_suffix):
        phase_sec = ((t_sec - T0) / period) % 1
        sidx = np.argsort(phase_sec)

        phase_sec = phase_sec[sidx]
        f_sorted = f_sec[sidx]
        e_sorted = e_sec[sidx] if e_sec is not None else np.ones_like(f_sorted)
        q_sorted = q_sec[sidx]

        good = (q_sorted == 0)
        flag = ~good

        phase_g = phase_sec[good]
        f_g = f_sorted[good]

        ms_g = marker_size_for_panel(len(phase_g), n_thresh=2000)
        ax.plot(
            phase_g, f_g, ".", color="k", alpha=0.3, ms=ms_g,
            label=(f"Good (Q=0), N={len(phase_g)}" if show_flagged_points else None)
        )

        if show_flagged_points and np.any(flag):
            phase_b = phase_sec[flag]
            f_b = f_sorted[flag]
            ms_b = marker_size_for_panel(len(phase_b), n_thresh=2000)
            flag_label = quality_summary(
                q_sorted[flag], frac_thresh=0.10, max_flags=2,
                denom="flagged", prefix="Flag"
            ) or "Flagged"
            ax.plot(
                phase_b, f_b, "x", color="tab:red", alpha=0.3, ms=ms_b,
                label=f"{flag_label}, N={len(phase_b)}"
            )

        ax.set_title(title_suffix, fontsize=14)
        ax.set_xlabel("Phase", fontsize=12)
        ax.set_ylabel("Norm. Flux", fontsize=12)
        ax.set_ylim(ymin_global, ymax_global)
        ax.grid()

        if show_flagged_points and np.any(flag):
            ax.legend(fontsize=12, loc="best")

    # ---------- FIGURES ----------
    if plot_by_sector:
        # FIGURE 1: periodogram + folded
        fig1, axes1 = plt.subplots(1, 3, figsize=(30, 6))
        fig1.suptitle(fig_title, fontsize=25, y=0.98)

        ax_pg = axes1[0]
        ax_pg.plot(freq, power, "-", lw=1)
        ax_pg.axvline(best_freq, ls="--", lw=2, label=f"Best: P={best_period:.3f} d")
        if np.isfinite(threshold):
            ax_pg.axhline(threshold, ls="--", alpha=0.7, label="Threshold")
        ax_pg.set_xlim(freq.min(), freq.max())
        ax_pg.set_title(f"LS | peak/median={peak_over_median:.2f}", fontsize=18)
        ax_pg.set_xlabel("Frequency [1/days]",fontsize=14)
        ax_pg.set_ylabel("Power",fontsize=14)
        ax_pg.grid()
        ax_pg.legend(fontsize=14)

        # Folded at P_best using GOOD points only (as in your LS)
        ax_fold = axes1[1]
        phase = ((time - T0) / best_period) % 1
        sidx = np.argsort(phase)
        phase, folded_flux = phase[sidx], flux[sidx]
        folded_err = flux_err[sidx] if flux_err is not None else np.ones_like(folded_flux)

        mfin = np.isfinite(phase) & np.isfinite(folded_flux) & np.isfinite(folded_err)
        phase, folded_flux, folded_err = phase[mfin], folded_flux[mfin], folded_err[mfin]

        if len(phase) >= 20:
            ms_fold = marker_size_for_panel(len(phase), n_thresh=2000)
            ax_fold.plot(phase, folded_flux, ".", alpha=0.3, ms=ms_fold)
            ax_fold.set_title(rf"Folded: $P={best_period:.3f}\pm{period_err:.3f}\,\mathrm{{d}}$",fontsize=18)
            ax_fold.set_xlabel("Phase",fontsize=14)
            ax_fold.set_ylabel("Norm. Flux",fontsize=14)
            ax_fold.grid()
            ax_fold.set_ylim(ymin_global, ymax_global)
        else:
            ax_fold.text(0.5, 0.5, "Too few points", ha="center", va="center")

        # Folded at 2*P_best
        ax_fold2 = axes1[2]
        p2 = 2.0 * best_period
        phase2 = ((time - T0) / p2) % 1
        sidx2 = np.argsort(phase2)
        phase2, folded_flux2 = phase2[sidx2], flux[sidx2]
        folded_err2 = flux_err[sidx2] if flux_err is not None else np.ones_like(folded_flux2)

        mfin2 = np.isfinite(phase2) & np.isfinite(folded_flux2) & np.isfinite(folded_err2)
        phase2, folded_flux2, folded_err2 = phase2[mfin2], folded_flux2[mfin2], folded_err2[mfin2]

        if len(phase2) >= 20:
            ms_fold = marker_size_for_panel(len(phase2), n_thresh=2000)
            ax_fold2.plot(phase2, folded_flux2, ".", alpha=0.3, ms=ms_fold)
            ax_fold2.set_title(rf"Folded: $P=2P_{{\rm best}}={p2:.3f}\,\mathrm{{d}}$",fontsize=18)
            ax_fold2.set_xlabel("Phase",fontsize=14)
            ax_fold2.set_ylabel("Norm. Flux",fontsize=14)
            ax_fold2.grid()
            ax_fold2.set_ylim(ymin_global, ymax_global)
        else:
            ax_fold2.text(0.5, 0.5, "Too few points", ha="center", va="center")

        plt.tight_layout()
        fig1.savefig(fig1_path, dpi=200)
        plt.close(fig1)

        # FIGURE 2: per-sector layout
        if use_sector_ls_panels:
            # cached sector LS arrays
            sec_ids = np.asarray(sector_cache["sectors"], int)
            pow2d = np.asarray(sector_cache["power"], float)
            best_freq_s = np.asarray(sector_cache["best_freq"], float)
            best_per_s = np.asarray(sector_cache["best_period"], float)
            perr_s = np.asarray(sector_cache["period_err"], float)
            thr_s = np.asarray(sector_cache["threshold"], float)
            peak_over_med_s = np.asarray(sector_cache["peak_over_median"], float)

            # common LS limits
            xlim_pg = (float(freq.min()), float(freq.max()))
            allp = pow2d[np.isfinite(pow2d)]
            y_max_pg = float(np.nanpercentile(allp, 99.9)) if allp.size else 1.0
            ylim_pg = (0.0, 1.05 * y_max_pg)

            fig_height = max(4.0, n_sectors * 3.2)
            fig2, axes2 = plt.subplots(
                nrows=n_sectors, ncols=4,
                figsize=(26, fig_height),
                gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.0]}
            )
            if n_sectors == 1:
                axes2 = np.array([axes2])
            axes2 = np.atleast_2d(axes2)

            suptithight = 1 - 0.05 / max(n_sectors, 1)
            fig2.suptitle(fig_title, fontsize=20, y=suptithight)

            sec_to_idx = {int(s): i for i, s in enumerate(sec_ids)}

            for i, (t_sec, f_sec, e_sec, q_sec, mi) in enumerate(zip(times, fluxes, fluxerrs, quals, meta)):
                sec = int(mi.get("sector", -1))
                j = sec_to_idx.get(sec, None)

                sec_str = f"S{sec}" if sec != -1 else "S-1"
                auth_str = mi.get("author", "unknown") or "unknown"
                flux_kind_str = mi.get("flux_kind", "UNKNOWN")

                nraw = int(mi.get("n_raw", len(t_sec)))
                ngood = int(mi.get("n_kept_good", int(np.sum(q_sec == 0))))
                nflag = int(mi.get("n_kept_flagged", int(np.sum(q_sec != 0))))

                # Column 1: raw
                ax_raw = axes2[i, 0]
                good = (q_sec == 0)
                flag = ~good
                ms_good = marker_size_for_panel(int(np.sum(good)), n_thresh=2000)
                ms_flag = marker_size_for_panel(int(np.sum(flag)), n_thresh=2000)

                ax_raw.plot(t_sec[good], f_sec[good], ".", color="k", markersize=ms_good, alpha=0.7,
                            label=f"Good (Q=0): {ngood}/{nraw}" if show_flagged_points else None)

                if show_flagged_points and np.any(flag):
                    flag_label = quality_summary(
                        q_sec[flag], frac_thresh=0.10, max_flags=2,
                        denom="flagged", prefix="Flag"
                    ) or "Flagged"
                    ax_raw.plot(
                        t_sec[flag], f_sec[flag], "x", color="tab:red",
                        markersize=ms_flag, alpha=0.7,
                        label=f"{flag_label}: {nflag}/{nraw}"
                    )

                soft_line = mi.get("quality_soft_summary_raw", "")
                if show_flagged_points:
                    title = f"{sec_str} | {auth_str} | {flux_kind_str} | Good={ngood}/{nraw} (Flagged={nflag})"
                    if soft_line:
                        title += "\n" + soft_line
                else:
                    title = f"{sec_str} | {auth_str} | {flux_kind_str} | Good={ngood}/{nraw}"

                ax_raw.set_title(title,fontsize=14)
                ax_raw.set_xlabel("Time [days]",fontsize=12)
                ax_raw.set_ylabel("Norm. Flux",fontsize=12)
                ax_raw.set_ylim(ymin_global, ymax_global)
                ax_raw.grid()

                if show_flagged_points and np.any(flag):
                    ax_raw.legend(fontsize=12, loc="best")

                # Column 2: cached LS
                ax_pg = axes2[i, 1]
                if j is not None:
                    ax_pg.plot(freq, pow2d[j], "-", lw=1)
                    ax_pg.axvline(best_freq_s[j], ls="--", lw=2, label=f"Pbest={best_per_s[j]:.3f} d")
                    if np.isfinite(thr_s[j]):
                        ax_pg.axhline(thr_s[j], ls="--", alpha=0.7, label="Threshold")
                    ax_pg.set_title(f"{sec_str} LS | peak/median={peak_over_med_s[j]:.2f}",fontsize=14)
                else:
                    ax_pg.text(0.5, 0.5, "No cached LS for this sector", ha="center", va="center")

                ax_pg.set_xlim(*xlim_pg)
                ax_pg.set_ylim(*ylim_pg)
                ax_pg.set_xlabel("Frequency [1/days]",fontsize=12)
                ax_pg.set_ylabel("Power",fontsize=12)
                ax_pg.grid()
                ax_pg.legend(fontsize=12, loc="best")

                # Column 3: folded at sector Pbest
                ax_f1 = axes2[i, 2]
                psec = float(best_per_s[j]) if j is not None else best_period
                perrsec = float(perr_s[j]) if (j is not None and np.isfinite(perr_s[j])) else np.nan
                _plot_fold(ax_f1, t_sec, f_sec, e_sec, q_sec, psec, T0,
                           rf"{sec_str} Folded: $P={psec:.3f}\pm{perrsec:.3f}\,\mathrm{{d}}$")

                # Column 4: folded at 2*sector Pbest
                ax_f2 = axes2[i, 3]
                _plot_fold(ax_f2, t_sec, f_sec, e_sec, q_sec, 2.0 * psec, T0,
                           rf"{sec_str} Folded: $P=2P_{{\rm best}}={2.0*psec:.3f}\,\mathrm{{d}}$")

            plt.tight_layout()
            fig2.savefig(fig2_path, dpi=200)
            plt.close(fig2)

        else:
            # cached per-sector fold figure without LS column
            fig_height = n_sectors * 4
            fig2, axes2 = plt.subplots(nrows=n_sectors, ncols=3, figsize=(18, fig_height),
                                       gridspec_kw={"width_ratios": [1, 1, 1]})
            if n_sectors == 1:
                axes2 = np.array([axes2])
            axes2 = np.atleast_2d(axes2)

            suptithight = 1 - 0.05 / max(n_sectors, 1)
            fig2.suptitle(fig_title, fontsize=20, y=suptithight)

            p2 = 2.0 * best_period

            for i, (t_sec, f_sec, e_sec, q_sec, mi) in enumerate(zip(times, fluxes, fluxerrs, quals, meta)):
                sec = int(mi.get("sector", -1))
                sec_str = f"S{sec}" if sec != -1 else "S-1"
                auth_str = mi.get("author", "unknown") or "unknown"

                nraw = int(mi.get("n_raw", len(t_sec)))
                ngood = int(mi.get("n_kept_good", int(np.sum(q_sec == 0))))
                nflag = int(mi.get("n_kept_flagged", int(np.sum(q_sec != 0))))

                ax_left = axes2[i, 0]
                good = (q_sec == 0)
                flag = ~good

                ms_good = marker_size_for_panel(int(np.sum(good)), n_thresh=2000)
                ms_flag = marker_size_for_panel(int(np.sum(flag)), n_thresh=2000)

                ax_left.plot(
                    t_sec[good], f_sec[good], ".", color="k", markersize=ms_good, alpha=0.7,
                    label=(f"Good (Q=0): {ngood}/{nraw}" if show_flagged_points else None)
                )

                if show_flagged_points and np.any(flag):
                    flag_label = quality_summary(
                        q_sec[flag], frac_thresh=0.10, max_flags=2,
                        denom="flagged", prefix="Flag"
                    ) or "Flagged"
                    ax_left.plot(
                        t_sec[flag], f_sec[flag], "x", color="tab:red",
                        markersize=ms_flag, alpha=0.7,
                        label=f"{flag_label}: {nflag}/{nraw}"
                    )

                soft_line = mi.get("quality_soft_summary_raw", "")

                if show_flagged_points:
                    title = f"{sec_str} | {auth_str} | Good={ngood}/{nraw} (Flagged={nflag})"
                    if soft_line:
                        title += "\n" + soft_line
                else:
                    title = f"{sec_str} | {auth_str} | Good={ngood}/{nraw}"

                ax_left.set_title(title, fontsize=14)

                if show_flagged_points and np.any(flag):
                    ax_left.legend(fontsize=12, loc="best")
                ax_left.set_xlabel("Time [days]",fontsize=12)
                ax_left.set_ylabel("Norm. Flux",fontsize=12)
                ax_left.set_ylim(ymin_global, ymax_global)
                ax_left.grid()

                _plot_fold(axes2[i, 1], t_sec, f_sec, e_sec, q_sec, best_period, T0,
                           rf"{sec_str} Folded: $P={best_period:.3f}\,\mathrm{{d}}$")
                _plot_fold(axes2[i, 2], t_sec, f_sec, e_sec, q_sec, p2, T0,
                           rf"{sec_str} Folded: $P=2P_{{\rm best}}={p2:.3f}\,\mathrm{{d}}$")

            plt.tight_layout()
            fig2.savefig(fig2_path, dpi=200)
            plt.close(fig2)

    else:
        # "Default mode" (no per-sector panels): reproduce your 2x2 figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ms_raw = marker_size_for_panel(len(time), n_thresh=2000)
        axes[0, 0].plot(time, flux, ".", markersize=ms_raw, alpha=0.7)
        axes[0, 0].set_title("Lightcurve (cached)",fontsize=14)
        axes[0, 0].set_xlabel("Time [days]",fontsize=12)
        axes[0, 0].set_ylabel("Normalized Flux",fontsize=12)
        axes[0, 0].grid()

        axes[0, 1].axvline(best_freq, color="m", ls="--", lw=2, label=f"Best Period: {best_period:.3f} d")
        axes[0, 1].plot(freq, power, "k-", lw=1)
        if np.isfinite(threshold):
            axes[0, 1].axhline(threshold, color="gray", ls="--", alpha=0.7, label="Threshold")
        axes[0, 1].set_xlim(freq.min(), freq.max())
        axes[0, 1].set_xlabel("Frequency [1/days]",fontsize=12)
        axes[0, 1].set_ylabel("Power",fontsize=12)
        axes[0, 1].set_title("Lomb–Scargle Periodogram (cached)",fontsize=14)
        axes[0, 1].grid()
        axes[0, 1].legend(fontsize=12)

        for i, p in enumerate([best_period, 2.0 * best_period]):
            phase = ((time - T0) / p) % 1
            sidx = np.argsort(phase)
            phase = phase[sidx]
            folded_flux = flux[sidx]
            ax = axes[1, i]
            ms_fold = marker_size_for_panel(len(phase), n_thresh=2000)
            ax.plot(phase, folded_flux, ".", alpha=0.3, ms=ms_fold)
            ax.set_title(f"Folded: P={p:.3f} d (cached)")
            ax.set_xlabel("Phase")
            ax.set_ylabel("Norm. Flux")
            ax.set_ylim(ymin_global, ymax_global)
            ax.grid()

        plt.tight_layout()
        fname = "per_sector_raw_periodogram_and_folded" if per_sector_periodograms else "per_sector_raw_and_folded"
        fig.savefig(os.path.join(star_dir, fname + ".png"), dpi=200)
        plt.close(fig)

    print(f"Recreated figures from cache in: {star_dir}")
    return True

def search_tess_lightcurves(
    tic_id=None, gaia_source_id=None, ra=None, dec=None,
    search_radius=2*u.arcmin, select_closest_on_sky=True
):
    """
    Returns a Lightkurve SearchResult for TESS lightcurves.

    Provide one of:
      - tic_id
      - gaia_source_id (Gaia DR3)
      - ra, dec in degrees

    If using gaia_source_id and ra/dec not provided, we resolve via Gaia TAP.
    """
    if tic_id is not None and str(tic_id).strip() != "":
        sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="SPOC")

        if len(sr) == 0:
            sr = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author="QLP")

        sr = sr[np.argsort(sr.table["sequence_number"])]

        return sr

    if (gaia_source_id is not None and str(gaia_source_id).strip() != "") and (ra is None or dec is None):
        ra, dec = gaia_source_id_to_radec(gaia_source_id)

    if ra is None or dec is None:
        raise ValueError("Must provide tic_id OR gaia_source_id OR both ra and dec.")

    coord = SkyCoord(float(ra), float(dec), unit="deg", frame="icrs")
    sr = lk.search_lightcurve(coord, mission="TESS", radius=search_radius, author="SPOC")

    if len(sr) == 0:
        sr = lk.search_lightcurve(coord, mission="TESS", radius=search_radius, author="QLP")

    sr = sr[np.argsort(sr.table["sequence_number"])]

    if select_closest_on_sky and len(sr) > 0 and hasattr(sr, "table"):
        tbl = sr.table
        if "distance" in tbl.colnames and "target_name" in tbl.colnames:
            i0 = int(np.argmin(tbl["distance"]))
            closest_name = tbl["target_name"][i0]
            mask = np.array(tbl["target_name"] == closest_name)
            sr = sr[mask]

    return sr

def marker_size_for_panel(n_points, n_thresh=2000, ms_small=4, ms_large=2):
    """
    If a panel has fewer than n_thresh points, use ms_small; otherwise ms_large.
    """
    try:
        n = int(n_points)
    except Exception:
        n = 0
    return ms_small if n < n_thresh else ms_large


def robust_mad_sigma(y):
    """Robust sigma estimate from MAD (scaled to match Gaussian std)."""
    y = np.asarray(y)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.nan
    med = np.nanmedian(y)
    return 1.4826 * np.nanmedian(np.abs(y - med))

def sector_scatter_metric(t, y, method="std"):
    """
    Scatter per sector.
    method:
      - "std": standard deviation of y (sensitive to real variability + trends)
      - "mad": robust sigma from MAD of y
      - "p2p": point-to-point scatter (more 'noise-like', less sensitive to slow trends)
    """
    y = np.asarray(y)
    t = np.asarray(t)
    m = np.isfinite(t) & np.isfinite(y)
    y = y[m]
    t = t[m]
    if y.size < 10:
        return np.nan

    if method == "std":
        return float(np.nanstd(y))
    if method == "mad":
        return float(robust_mad_sigma(y))
    if method == "p2p":
        # robust point-to-point scatter: MAD of first differences / sqrt(2)
        dy = np.diff(y)
        return float(robust_mad_sigma(dy) / np.sqrt(2))

    raise ValueError(f"Unknown method='{method}'")

def focus_fmax_by_excess_power(freq, power, floor_power, q=0.995):
    """
    Choose fmax so that q fraction of the cumulative "excess power"
    (power - floor_power, clipped at 0) is contained below fmax.
    """
    f = np.asarray(freq, float)
    p = np.asarray(power, float)

    good = np.isfinite(f) & np.isfinite(p)
    if np.sum(good) < 100 or (not np.isfinite(floor_power)) or floor_power <= 0:
        return float(np.nanmax(f))

    excess = np.clip(p - floor_power, 0, None)
    cum = np.cumsum(excess[good])
    if cum.size == 0 or cum[-1] <= 0:
        return float(np.nanmax(f))

    target = q * cum[-1]
    idx = int(np.searchsorted(cum, target))
    idx = min(max(idx, 0), np.sum(good) - 1)
    f_good = f[good]
    return float(f_good[idx])

# -----------------------------
# QUALITY flag decoding + summaries
# -----------------------------

def _tess_quality_strings():
    """
    Try Lightkurve's official mapping first. Fall back to a minimal SPOC-ish map.
    Keys are BITMASK VALUES (1,2,4,8,...), not bit indices.
    """
    # Try to use Lightkurve's official bit->string map
    strings = {}
    try:
        from lightkurve.utils import TessQualityFlags
        strings = dict(getattr(TessQualityFlags, "STRINGS", {}) or {})
        strings[1073741824] = "QLP quality flag"  # 2**30
    except Exception:
        strings = {}

    # Minimal fallback
    if not strings:
        strings = {
            1: "Attitude tweak",
            2: "Safe mode",
            4: "Coarse point",
            8: "Earth point",
            16: "Argabrightening",
            32: "Reaction wheel desat",
            64: "Cosmic ray (aperture)",
            128: "Manual exclude",
            256: "Discontinuity corrected",
            512: "Impulsive outlier",
            1024: "Collateral cosmic ray",
            2048: "Straylight (predicted)",
            4096: "Scattered light exclude",
            8192: "Straylight (extended)",
            16384: "Bad calibration exclude",
            32768: "Insufficient targets",
            1073741824: "QLP quality flag",
        }

    # If QLP uses a high-level "quality" sentinel:
    strings.setdefault(1 << 30, "QLP quality flag")

    # Remove any accidental zero entry
    strings.pop(0, None)
    return strings


_TESS_QUALITY_STRINGS = _tess_quality_strings()

# Bit *indices* that are usually "hard discard" in practice.
# These indices correspond to values 2**(bit-1).
_HARD_BIT_IDXS = {2, 3, 4, 6, 8, 13, 15, 16, 30}
HARD_QUALITY_MASK = sum(1 << (b - 1) for b in _HARD_BIT_IDXS)


def quality_bit_fractions(q, denom="all", ignore_mask=0):
    """
    Return list of (frac, name, bit_value) sorted descending by frac.

    denom:
      - "all": fractions are w.r.t. all cadences in q
      - "flagged": fractions are w.r.t. only nonzero-quality cadences
    ignore_mask: bitmask values to ignore (e.g., HARD_QUALITY_MASK for "soft-only")
    """
    q = np.asarray(q, dtype=np.int64)
    if q.size == 0:
        return []

    base = q[q != 0] if denom == "flagged" else q
    if base.size == 0:
        return []

    hits = []
    for bit_val, name in _TESS_QUALITY_STRINGS.items():
        bit_val = int(bit_val)
        if bit_val == 0 or (ignore_mask & bit_val):
            continue
        frac = float(np.mean((base & bit_val) != 0))
        if frac > 0:
            hits.append((frac, str(name), bit_val))

    hits.sort(reverse=True, key=lambda x: x[0])
    return hits


def quality_summary(q, frac_thresh=0.10, max_flags=3, denom="all", ignore_mask=0, prefix="Q", max_len=90):
    """
    Build compact string like:
      "Q: Impulsive outlier 22%, Argabrightening 15%"

    Designed for titles/legends.
    """
    hits = [h for h in quality_bit_fractions(q, denom=denom, ignore_mask=ignore_mask) if h[0] >= frac_thresh]
    if not hits:
        return ""

    parts = [f"{name} {frac*100:.0f}%" for frac, name, _ in hits[:max_flags]]
    s = f"{prefix}: " + ", ".join(parts)
    if len(s) > max_len:
        s = s[:max_len - 1] + "…"
    return s


def quality_present_lists(q):
    """
    For metrics: return (bit_values_present, flag_names_present)
    considering ANY presence (>0 frac) across all cadences.
    """
    hits = quality_bit_fractions(q, denom="all", ignore_mask=0)
    bits = [int(bit) for frac, _, bit in hits if frac > 0]
    names = [str(name) for frac, name, _ in hits if frac > 0]
    return bits, names


def quality_summary_for_title(q, frac_thresh=0.30, max_flags=2, max_len=90):
    """
    Build a short summary of common TESS QUALITY bits in this sector.

    Example: "Q: Scattered Light Exclude 98%, Straylight 45%"

    frac_thresh: show flags present in >= this fraction of cadences
    max_flags  : max number of flags to list (keeps titles short)
    """
    q = np.asarray(q, dtype=np.int64)
    if q.size == 0:
        return ""

    # If everything is zero, nothing to report
    if np.nanmax(q) == 0:
        return ""

    # Try to use Lightkurve's official bit->string map
    strings = {}
    try:
        from lightkurve.utils import TessQualityFlags
        strings = dict(getattr(TessQualityFlags, "STRINGS", {}) or {})
        strings[1073741824] = "QLP quality flag"  # 2**30


    except Exception:
        strings = {}

    # Fallback if STRINGS isn't available for some reason
    if not strings:
        vals, cnts = np.unique(q, return_counts=True)
        i = int(np.argmax(cnts))
        return f"Q: mode={int(vals[i])} ({100*cnts[i]/q.size:.0f}%)"

    hits = []
    for bit, name in strings.items():
        try:
            bit = int(bit)
        except Exception:
            continue
        if bit == 0:
            continue

        frac = float(np.mean((q & bit) != 0))
        if frac >= frac_thresh:
            hits.append((frac, str(name)))

    hits.sort(reverse=True, key=lambda x: x[0])
    if not hits:
        return ""

    parts = [f"{name} {frac*100:.0f}%" for frac, name in hits[:max_flags]]
    s = "Q: " + ", ".join(parts)

    if len(s) > max_len:
        s = s[:max_len - 1] + "…"
    return s

def compute_gls_metrics(time, flux, flux_err, freq):
    """
    Compute GLS Lomb-Scargle power on a provided freq grid, pick best peak,
    compute FWHM period error, and compute peak "goodness" ratios.
    Returns a dict with metrics + power array.
    """
    out = {
        "power": None,
        "threshold": np.nan,
        "n_peaks": 0,
        "note": "",
        "best_idx": np.nan,
        "best_freq": np.nan,
        "best_period_days": np.nan,
        "period_err_days": np.nan,
        "best_power": np.nan,
        "best_power_over_median": np.nan,
        "power_floor_flat": np.nan,
        "best_power_over_flat_floor": np.nan,
        "power_floor_local": np.nan,
        "best_power_over_local_floor": np.nan,
        "f_nonflat_max": np.nan,
        "floor_log10": np.nan,
        "floor_mad_log10": np.nan,
    }

    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    if flux_err is not None:
        flux_err = np.asarray(flux_err, float)

    good = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        good &= np.isfinite(flux_err) & (flux_err > 0)

    time = time[good]
    flux = flux[good]
    if flux_err is not None:
        flux_err = flux_err[good]

    if time.size < 20:
        out["power"] = np.full_like(freq, np.nan, dtype=float)
        out["note"] = "(too few points)"
        return out

    # GLS (chi2/standard) like your combined case
    try:
        dy = None
        if flux_err is not None:
            dy = flux_err.copy()
            bad = (~np.isfinite(dy)) | (dy <= 0)
            if np.any(bad):
                g = dy[~bad]
                dy = None if g.size == 0 else np.where(bad, np.nanmedian(g), dy)

        ls = LombScargle(time, flux, dy) if dy is not None else LombScargle(time, flux)
        power = ls.power(freq, method="chi2", normalization="standard")
    except Exception:
        power = np.full_like(freq, np.nan, dtype=float)

    if not np.any(np.isfinite(power)):
        power = np.zeros_like(freq, dtype=float)

    out["power"] = power

    pmed = float(np.nanmedian(power))
    thr = pmed * 1.2
    out["threshold"] = float(thr)

    peaks, _ = find_peaks(power, height=thr)
    out["n_peaks"] = int(len(peaks))

    if len(peaks) == 0:
        best_idx = int(np.nanargmax(power))
        note = "(fallback: max power)"
    else:
        best_idx = int(peaks[np.nanargmax(power[peaks])])
        note = "(best significant peak)"
    out["note"] = note

    best_freq = float(freq[best_idx])
    best_period = float(1.0 / best_freq) if best_freq > 0 else np.nan

    # reject P > baseline
    T = float(np.nanmax(time) - np.nanmin(time))
    if np.isfinite(best_period) and np.isfinite(T) and (best_period > T) and (T > 0):
        sorted_idx = np.argsort(power)[::-1]
        for idx in sorted_idx:
            cand = 1.0 / float(freq[idx])
            if cand <= T:
                best_idx = int(idx)
                best_freq = float(freq[best_idx])
                best_period = float(cand)
                note = "(corrected: within baseline)"
                out["note"] = note
                break

    out["best_idx"] = int(best_idx)
    out["best_freq"] = best_freq
    out["best_period_days"] = best_period

    best_power = float(power[best_idx])
    out["best_power"] = best_power
    out["best_power_over_median"] = float(best_power / pmed) if (np.isfinite(pmed) and pmed > 0) else np.nan

    # Flat floor diagnostics
    floor_power, floor_lp, floor_mad_lp, f_nonflat_max, lp_smooth, thr_lp = \
        estimate_flat_floor_and_nonflat_max(freq, power, hi_frac=0.20, smooth_frac=0.02, k=3.0)

    out["power_floor_flat"] = float(floor_power) if np.isfinite(floor_power) else np.nan
    out["floor_log10"] = float(floor_lp) if np.isfinite(floor_lp) else np.nan
    out["floor_mad_log10"] = float(floor_mad_lp) if np.isfinite(floor_mad_lp) else np.nan
    out["f_nonflat_max"] = float(f_nonflat_max) if np.isfinite(f_nonflat_max) else np.nan
    out["best_power_over_flat_floor"] = float(best_power / floor_power) if (np.isfinite(floor_power) and floor_power > 0) else np.nan

    # Local floor around best peak
    local_floor = local_periodogram_floor(power, best_idx, half_width=3000, exclude=200)
    out["power_floor_local"] = float(local_floor) if np.isfinite(local_floor) else np.nan
    out["best_power_over_local_floor"] = float(best_power / local_floor) if (np.isfinite(local_floor) and local_floor > 0) else np.nan

    # FWHM error (same method as you used)
    half_max = best_power / 2.0
    left_idxs = np.where(power[:best_idx] < half_max)[0]
    left_idx = int(left_idxs[-1]) if left_idxs.size > 0 else 0
    right_idxs = np.where(power[best_idx:] < half_max)[0]
    right_idx = int(best_idx + right_idxs[0]) if right_idxs.size > 0 else (len(power) - 1)

    fwhm = float(freq[right_idx] - freq[left_idx])
    out["period_err_days"] = float(fwhm / (best_freq ** 2)) if np.isfinite(best_freq) and best_freq > 0 else np.nan

    return out

def infer_columns(df):
    """
    Best-effort detection of TIC and RA/Dec columns (case-insensitive).
    Returns: (tic_col, ra_col, dec_col)
    """
    cols = {c.lower(): c for c in df.columns}

    # TIC candidates (extend as needed)
    tic_candidates = ["tic", "tic_id", "ticid", "tess_tic", "id", "tic_num", "tess_ID", "TESS_ID"]
    gaia_candidates = ["source_id", "gaia_source_id", "gaia_sourceid", "gaia_id", "gaia_dr3_source_id", "dr3_source_id"]
    ra_candidates  = ["ra", "radeg", "ra_deg"]
    dec_candidates = ["dec", "dedeg", "decdeg", "dec_deg", "de_deg"]


    tic_col = next((cols[k] for k in tic_candidates if k in cols), None)
    gaia_col = next((cols[k] for k in gaia_candidates if k in cols), None)
    ra_col  = next((cols[k] for k in ra_candidates if k in cols), None)
    dec_col = next((cols[k] for k in dec_candidates if k in cols), None)


    return tic_col, gaia_col, ra_col, dec_col


# -----------------------------
# Main analyzer
# -----------------------------
def analyze_tess_lightcurve(
    tic_id=None, gaia_source_id=None, ra=None, dec=None,
    output_dir="TESSPlots",
    search_radius=2*u.arcmin,
    interactive=True,
    plot_by_sector_default=True,
    use_default_freq_range=True,
    fmin=None, fmax=None,
    return_metrics=False,
    batch_row_index=None,
    per=None,          # Gao P_rot (column "Per")
    psini=None,        # Gaia P/sini (column "Psini")
    gaia_period=None,  # old "gaia period" (Pasquale)
    gmag=None,
    flux_mode="default",
    per_sector_periodograms=False,
    plot_output_layout="by_star",
    cache_policy="auto",
    show_flagged_points=False,
):

    # --- label (star name) ---

    if tic_id is not None and str(tic_id).strip() != "":
        label = f"TIC {tic_id}"
    elif gaia_source_id is not None and str(gaia_source_id).strip() != "":
        label = f"Gaia DR3 {gaia_source_id}"
    else:
        label = f"RA={ra}, Dec={dec}"

    target_id = sanitize_target_id(tic_id=tic_id, gaia_source_id=gaia_source_id, ra=ra, dec=dec)

    fig_title = build_fig_suptitle(
        label,
        per=per,
        psini=psini,
        gaia_period=gaia_period,
        gmag=gmag
    )

    metrics = {
        "status": "started",
        "error": "",
        "label": label,
        "target_id": target_id,
        "tic_id": str(tic_id) if tic_id is not None else "",
        "gaia_source_id": str(gaia_source_id) if gaia_source_id is not None else "",
        "ra_deg": float(ra) if ra not in [None, ""] else np.nan,
        "dec_deg": float(dec) if dec not in [None, ""] else np.nan,
        "flux_mode_requested": str(flux_mode),

        # search / sector stats
        "authors_used": "",
        "flux_kinds_used": "",
        "n_search_results": np.nan,
        "n_downloaded_sectors": np.nan,
        "n_usable_sectors": np.nan,
        "sector_numbers_downloaded": "",
        "raw_points_per_sector": "",
        "kept_points_per_sector": "",
        "kept_frac_per_sector": "",
        "sector_ls_json": "",

        # time/flux stats
        "time_min_btjd": np.nan,
        "time_max_btjd": np.nan,
        "baseline_days": np.nan,
        "n_points_total": np.nan,
        "flux_mean": np.nan,
        "flux_std": np.nan,

        # LS grid + power stats
        "fmin": np.nan,
        "fmax": np.nan,
        "n_freq": np.nan,
        "power_min": np.nan,
        "power_max": np.nan,
        "power_median": np.nan,
        "n_peaks": np.nan,
        "threshold": np.nan,

        # best period stats
        "best_idx": np.nan,
        "best_freq": np.nan,
        "best_period_days": np.nan,
        "period_err_days": np.nan,
        "T0_btjd": np.nan,
        "note": "",

        # peak height / relative height
        "best_power": np.nan,
        "best_power_over_median": np.nan,
        "power_floor_flat": np.nan,
        "best_power_over_flat_floor": np.nan,
        "power_floor_local": np.nan,
        "best_power_over_local_floor": np.nan,

        # "non-flat" plot region diagnostic
        "f_nonflat_max": np.nan,
        "floor_log10": np.nan,
        "floor_mad_log10": np.nan,

        # outputs
        "outdir": "",
        "cleaned_lightcurve_csv": "",
    }

    def _finish(status, error_msg=""):
        metrics["status"] = status
        metrics["error"] = error_msg
        return metrics if return_metrics else None

    # Create output dirs/paths early so cache can be checked before download/LS
    star_dir, fig1_path, fig2_path = resolve_plot_paths(output_dir, target_id, layout=plot_output_layout)

    cache_policy = (cache_policy or "auto").strip().lower()
    have_cache = _has_core_cache(star_dir)

    if cache_policy in ("auto", "use") and have_cache:
        print(f"Using cached products for {label} -> skipping download + LS recomputation.")
        try:
            fill_metrics_from_cache(metrics, star_dir, per_sector_periodograms=per_sector_periodograms)
            try:
                ok = recreate_figures_from_cached(
                    star_dir=star_dir,
                    fig1_path=fig1_path,
                    fig2_path=fig2_path,
                    fig_title=fig_title,
                    interactive=interactive,
                    plot_by_sector_default=plot_by_sector_default,
                    per_sector_periodograms=per_sector_periodograms,
                    show_flagged_points=show_flagged_points,
                )
            except Exception as e:
                metrics["error"] = f"cache_plot_failed: {e}"
                ok = False

            return _finish("ok_cached" if ok else "ok_cached_plot_failed", metrics.get("error", ""))

        except Exception as e:
            if cache_policy == "use":
                return _finish("cache_metrics_failed", str(e))
            else:
                print(f"Cache read failed for {label}, falling back to fresh download: {e}")

        # 2) (optional) recreate figures; do NOT fail the table if plotting fails
        try:
            ok = recreate_figures_from_cached(
                star_dir=star_dir,
                fig1_path=fig1_path,
                fig2_path=fig2_path,
                fig_title=fig_title,
                interactive=interactive,
                plot_by_sector_default=plot_by_sector_default,
                per_sector_periodograms=per_sector_periodograms,
                show_flagged_points=show_flagged_points,
            )
        except Exception as e:
            metrics["error"] = f"cache_plot_failed: {e}"
            ok = False

        return _finish("ok_cached" if ok else "ok_cached_plot_failed", metrics.get("error", ""))

    if cache_policy == "use" and not have_cache:
        return _finish("cache_missing", f"cache_policy='use' but missing cache in {star_dir}")

    print(f"Downloading TESS lightcurve for {label}...")

    try:
        search_result = search_tess_lightcurves(
            tic_id=tic_id,
            gaia_source_id=gaia_source_id,
            ra=ra, dec=dec,
            search_radius=search_radius,
            select_closest_on_sky=True
        )

    except Exception as e:
        print(f"Search failed for {label}: {e}")
        return _finish("search_failed", str(e))

    if len(search_result) == 0:
        print(f"No TESS data found for {label}")
        metrics["n_search_results"] = 0
        return _finish("no_tess_data", "")

    metrics["n_search_results"] = int(len(search_result))
    sector_numbers = None
    if hasattr(search_result, "table"):
        tbl = search_result.table
        if "sequence_number" in tbl.colnames:
            sector_numbers = [int(x) if str(x).strip() != "" else None for x in tbl["sequence_number"]]

    tbl = search_result.table if hasattr(search_result, "table") else None

    times, fluxes, fluxerrs, quals = [], [], [], []
    meta = []  # one entry per successfully downloaded LC

    for k in range(len(search_result)):
        sr = search_result[k]

        # --- metadata from the search table (if present) ---
        sec = None
        auth = ""
        exptime = None
        if tbl is not None:
            if "sequence_number" in tbl.colnames:
                try:
                    sec = int(tbl["sequence_number"][k])
                except Exception:
                    sec = None
            if "author" in tbl.colnames:
                try:
                    auth = str(tbl["author"][k])
                except Exception:
                    auth = ""
            if "exptime" in tbl.colnames:
                try:
                    exptime = float(tbl["exptime"][k])
                except Exception:
                    exptime = None

        try:
            lc = safe_download(sr)
            if lc is None:
                continue


            flux_kind, flux_col, flux_err_col = pick_flux_columns(lc, author=auth, flux_mode=flux_mode)

            # If we found a specific column, explicitly select it so lc.flux matches what we report
            try:
                if flux_col is not None:
                    if flux_err_col is not None:
                        lc = lc.select_flux(flux_col, flux_err_col)
                    else:
                        lc = lc.select_flux(flux_col)
            except Exception:
                # If select_flux fails for any reason, continue with the default lc.flux
                pass

            try:
                lc = lc.remove_nans()

                # drop inf/-inf (remove_nans does NOT remove inf)
                finite = np.isfinite(lc.flux.value)
                if getattr(lc, "flux_err", None) is not None:
                    finite &= np.isfinite(lc.flux_err.value)
                lc = lc[finite]

                # force writable backing arrays (fixes "output array is read-only")
                lc.flux = lc.flux.copy()
                if getattr(lc, "flux_err", None) is not None:
                    lc.flux_err = lc.flux_err.copy()

                lc = lc.normalize()

            except Exception as e:
                print(f"not normalizing a sector due to error: {e}")
                # If remove_nans().normalize fails for any reason, continue with the default lc.flux
                pass

            t = lc.time.value
            if np.nanmedian(t) > 10000:  # looks like full BJD
                t = t - 2457000

            fvals = lc.flux.value
            if hasattr(lc, "flux_err") and lc.flux_err is not None:
                evals = lc.flux_err.value
            else:
                evals = np.zeros_like(fvals)


            try:
                if hasattr(lc, "quality") and lc.quality is not None:
                    qvals = np.asarray(lc.quality, dtype=np.int64)
                elif "quality" in getattr(lc, "colnames", []):
                    qvals = np.asarray(lc["quality"], dtype=np.int64)
                else:
                    qvals = np.zeros_like(t, dtype=np.int64)
            except Exception:
                qvals = np.zeros_like(t, dtype=np.int64)

            times.append(t)
            fluxes.append(fvals)
            fluxerrs.append(evals)
            quals.append(qvals)

            meta.append({
                "sector": sec,
                "author": auth,
                "exptime": exptime,
                "n_raw": int(len(t)),
                "k_search": int(k),
                "flux_kind": flux_kind,
                "flux_col": str(flux_col) if flux_col else "",
                "flux_err_col": str(flux_err_col) if flux_err_col else "",
            })


        except Exception as e:
            print(f"Skipping a sector due to error: {e}")

    metrics["n_downloaded_sectors"] = int(len(times))
    if sector_numbers is not None:
        # sector_numbers corresponds to the *search_result rows*, but we may have skipped downloads
        # so we won't try to perfectly align. We'll store the whole list as a hint.
        metrics["sector_numbers_downloaded"] = json.dumps(sector_numbers)
        metrics["authors_used"] = json.dumps(sorted({m.get("author", "") for m in meta if m.get("author", "")}))
        metrics["flux_kinds_used"] = json.dumps(
            sorted({m.get("flux_kind", "") for m in meta if m.get("flux_kind", "")}))

    if len(times) == 0:
        print(f"No usable lightcurves for {label}")
        return _finish("no_usable_lightcurves", "")

    n_sectors = len(times)
    print(f"Found {n_sectors} usable sectors for {label}.")

    if interactive:
        sector_choice = input("Plot each sector individually? [y/N]: ").strip().lower()
        plot_by_sector = sector_choice in ["y", "yes"]
    else:
        plot_by_sector = bool(plot_by_sector_default)

    raw_counts = [len(t) for t in times]

    # --- per-sector filtering ---
    KEEP_FRAC_MIN = 0.50  # drop if >50% points removed

    times_f, fluxes_f, fluxerrs_f, quals_f = [], [], [], []
    kept_counts = []  # per original sector (pre-drop)
    kept_fracs = []  # per original sector (pre-drop)
    drop_by_frac_idx = []  # indices in the original downloaded list

    print("Data counts per sector after filtering:")
    meta_f = []

    for i, (t_sec, f_sec, e_sec, q_sec, mi) in enumerate(zip(times, fluxes, fluxerrs, quals, meta)):
        n_raw = int(len(t_sec))  # raw after remove_nans().normalize()

        median_flux = np.median(f_sec)
        mad_flux = 1.4826 * np.median(np.abs(f_sec - median_flux))
        K_UP = 5.0  # keep this strict-ish to remove upward spikes
        K_DOWN = 500.0  # allow deeper dips (less aggressive below bulk)
        min_flux = 0.4

        mask = (
                (f_sec <= median_flux + K_UP * mad_flux) &
                (f_sec >= median_flux - K_DOWN * mad_flux) &
                (f_sec >= min_flux)
        )
        finite_mask = mask & np.isfinite(t_sec) & np.isfinite(f_sec) & np.isfinite(e_sec)

        t_sec, f_sec, e_sec, q_sec = t_sec[finite_mask], f_sec[finite_mask], e_sec[finite_mask], q_sec[finite_mask]

        n_after_outlier = int(len(t_sec))
        mi["n_outliers_removed"] = int(n_raw - n_after_outlier)  # removed by your flux/outlier mask

        good_q = (q_sec == 0)
        mi["n_kept_good"] = int(np.sum(good_q))
        mi["n_kept_flagged"] = int(np.sum(~good_q))

        # optional: these are referenced later in titles/metrics; set them here too
        bits, names = quality_present_lists(q_sec)
        mi["quality_bits_present_raw"] = bits
        mi["quality_names_present_raw"] = names
        mi["quality_soft_summary_raw"] = quality_summary(
            q_sec, frac_thresh=0.30, max_flags=2, denom="all",
            ignore_mask=HARD_QUALITY_MASK, prefix="Soft"
        )

        n_kept = int(len(t_sec))
        frac = (n_kept / n_raw) if n_raw > 0 else 0.0

        kept_counts.append(n_kept)
        kept_fracs.append(frac)

        sec = mi.get("sector", None)
        sec_str = f"S{sec}" if sec is not None else f"Idx{i + 1}"

        print(f"{sec_str}: {n_kept} / {n_raw} points kept (frac={frac:.2f})")

        # record in meta
        mi["n_raw"] = n_raw
        mi["n_kept"] = n_kept
        mi["kept_frac"] = frac

        # HARD DROP: too much removed
        if frac < KEEP_FRAC_MIN:
            drop_by_frac_idx.append(i)
            print(f"  -> Dropping {sec_str}: kept_frac={frac:.2f} < {KEEP_FRAC_MIN:.2f}")
            continue

        if n_kept > 0:
            times_f.append(t_sec)
            fluxes_f.append(f_sec)
            fluxerrs_f.append(e_sec)
            quals_f.append(q_sec)
            meta_f.append(mi)

    times, fluxes, fluxerrs, quals = times_f, fluxes_f, fluxerrs_f, quals_f
    meta = meta_f

    metrics["raw_points_per_sector"] = json.dumps([int(x) for x in raw_counts])
    metrics["kept_points_per_sector"] = json.dumps([int(x) for x in kept_counts])
    metrics["kept_frac_per_sector"] = json.dumps([float(x) for x in kept_fracs])
    metrics["note"] = metrics.get("note", "")  # keep any prior note

    # -----------------------------
    # Drop sectors with anomalous scatter (MAD of sector-MAD)
    # -----------------------------
    DROP_OUTLIER_SECTORS = True
    SCATTER_METHOD = "mad"  # sector metric: MAD-based sigma of flux within sector
    SCATTER_SIGMA_CLIP = 4.0  # how aggressive
    CLIP_IN_LOG = True  # recommended: scatter spans orders of magnitude

    if DROP_OUTLIER_SECTORS and len(fluxes) >= 3:
        # per-sector scatter (this is the sector "MAD scatter" if method="mad")
        scat = np.array([
            sector_scatter_metric(t, f, method=SCATTER_METHOD)
            for t, f in zip(times, fluxes)
        ], dtype=float)

        # sanitize
        scat[~np.isfinite(scat)] = np.nan
        scat[scat <= 0] = np.nan

        valid = np.isfinite(scat)
        if np.sum(valid) >= 3:

            if CLIP_IN_LOG:
                v = np.log10(scat[valid])
                med = np.nanmedian(v)
                mad = 1.4826 * np.nanmedian(np.abs(v - med))  # MAD across sectors (in log space)
                if not np.isfinite(mad) or mad == 0:
                    mad = np.nanstd(v)

                z = (np.log10(scat) - med) / mad
            else:
                v = scat[valid]
                med = np.nanmedian(v)
                mad = 1.4826 * np.nanmedian(np.abs(v - med))  # MAD across sectors (linear)
                if not np.isfinite(mad) or mad == 0:
                    mad = np.nanstd(v)

                z = (scat - med) / mad

            keep = np.isfinite(z) & (np.abs(z) <= SCATTER_SIGMA_CLIP)

            drop_idx = np.where(~keep)[0]
            if drop_idx.size > 0:
                print("\nDropping sectors due to anomalous scatter (MAD-of-sectors criterion):")
                for j in drop_idx:
                    mi = meta[j]
                    sec = mi.get("sector", None)
                    auth = mi.get("author", "unknown")
                    print(f"  idx={j + 1:02d} sector={sec} author={auth} "
                          f"N={len(fluxes[j])} scatter({SCATTER_METHOD})={scat[j]:.3e} z={z[j]:.2f}")

            # apply mask
            times = [t for t, k in zip(times, keep) if k]
            fluxes = [f for f, k in zip(fluxes, keep) if k]
            fluxerrs = [e for e, k in zip(fluxerrs, keep) if k]
            quals = [q for q, k in zip(quals, keep) if k]
            meta = [m for m, k in zip(meta, keep) if k]

    metrics["n_usable_sectors"] = int(len(times))
    metrics["raw_points_per_sector"] = json.dumps([int(x) for x in raw_counts])
    metrics["kept_points_per_sector"] = json.dumps([int(x) for x in kept_counts])
    fracs = [(kc / rc if rc > 0 else np.nan) for kc, rc in zip(kept_counts, raw_counts)]
    metrics["kept_frac_per_sector"] = json.dumps(fracs)

    n_sectors = len(times)

    metrics["n_usable_sectors"] = int(n_sectors)


    if n_sectors == 0:
        print("No data survived filtering!")
        return _finish("no_data_after_filtering", "")

    time_all = np.concatenate(times)
    flux_all = np.concatenate(fluxes)
    flux_err_all = np.concatenate(fluxerrs)
    qual_all = np.concatenate(quals)

    good_all = (qual_all == 0)

    # These are the ONLY arrays used for Lomb–Scargle
    time = time_all[good_all]
    flux = flux_all[good_all]
    flux_err = flux_err_all[good_all]

    # Keep flagged arrays for plotting overlays
    time_flag = time_all[~good_all]
    flux_flag = flux_all[~good_all]
    flux_err_flag = flux_err_all[~good_all]
    qual_flag = qual_all[~good_all]

    metrics["n_points_total_all"] = int(len(time_all))
    metrics["n_points_good_total"] = int(np.sum(good_all))
    metrics["n_points_flagged_total"] = int(np.sum(~good_all))

    metrics["time_min_btjd"] = float(time.min())
    metrics["time_max_btjd"] = float(time.max())
    metrics["n_points_total"] = int(len(time))
    metrics["flux_mean"] = float(np.mean(flux))
    metrics["flux_std"] = float(np.std(flux))


    if np.all(flux_err == 0):
        flux_err = None

    print(f"Time range: {time.min():.2f} - {time.max():.2f}, N={len(time)}")
    print(f"Flux stats: mean={np.mean(flux):.3e}, std={np.std(flux):.3e}")

    # -----------------------------
    # Lomb–Scargle frequency grid (FIXED by default)
    # -----------------------------
    T = float(time.max() - time.min())
    metrics["baseline_days"] = float(T)

    def _baseline_defaults(T_):
        fmin_b = max(1.0 / T_, 1e-4) if (np.isfinite(T_) and T_ > 0) else 1e-4
        fmax_b = 4.0
        return fmin_b, fmax_b

    if interactive:
        choice = input("Use default frequency range (fixed)? [Y/n]: ").strip().lower()
        if choice in ["n", "no"]:
            try:
                fmin_in = float(input("Enter minimum frequency [1/days]: ").strip())
                fmax_in = float(input("Enter maximum frequency [1/days]: ").strip())
                fmin = fmin_in
                fmax = fmax_in
            except Exception as e:
                print(f"Invalid input ({e}). Using defaults.")
                fmin, fmax = (DEFAULT_FMIN, DEFAULT_FMAX) if use_default_freq_range else _baseline_defaults(T)
        else:
            fmin, fmax = (DEFAULT_FMIN, DEFAULT_FMAX) if use_default_freq_range else _baseline_defaults(T)

    else:
        if fmin is None or fmax is None:
            fmin, fmax = (DEFAULT_FMIN, DEFAULT_FMAX) if use_default_freq_range else _baseline_defaults(T)
        else:
            fmin, fmax = float(fmin), float(fmax)

            # sanitize
            if not np.isfinite(fmin) or fmin <= 0:
                fmin = DEFAULT_FMIN if use_default_freq_range else _baseline_defaults(T)[0]
            if not np.isfinite(fmax) or fmax <= fmin:
                fmax = DEFAULT_FMAX if use_default_freq_range else _baseline_defaults(T)[1]

    print(f"Frequency range: fmin={fmin:.6f}, fmax={fmax:.6f} (fixed n_freq={N_FREQ_DEFAULT})")

    # FIXED grid length for all stars (same n_freq), and same range if defaults are used
    freq = np.linspace(fmin, fmax, N_FREQ_DEFAULT, dtype=float)

    metrics["n_freq"] = int(len(freq))
    metrics["fmin"] = float(freq.min())
    metrics["fmax"] = float(freq.max())

    metrics["n_freq"] = int(len(freq))
    metrics["fmax"] = float(freq.max())  # record actual achieved max (will be ~4)
    metrics["fmin"] = float(fmin)

    sector_ls = []
    if per_sector_periodograms:
        for (t_sec, f_sec, e_sec, q_sec, mi) in zip(times, fluxes, fluxerrs, quals, meta):
            good = (q_sec == 0) & np.isfinite(t_sec) & np.isfinite(f_sec)
            tg = t_sec[good]
            fg = f_sec[good]
            eg = e_sec[good] if (e_sec is not None and np.any(e_sec)) else None

            sm = compute_gls_metrics(tg, fg, eg, freq)

            sector_ls.append({
                "sector": (mi.get("sector") if mi.get("sector") is not None else -1),
                "author": (mi.get("author") or "unknown"),
                "flux_kind": (mi.get("flux_kind") or "UNKNOWN"),
                "best_period_days": sm["best_period_days"],
                "period_err_days": sm["period_err_days"],
                "best_power": sm["best_power"],
                "best_power_over_median": sm["best_power_over_median"],
                "best_power_over_local_floor": sm["best_power_over_local_floor"],
            })

        metrics["sector_ls_json"] = json.dumps(sector_ls)

    # -----------------------------
    # GLS Lomb–Scargle (χ²-based)
    # -----------------------------
    try:
        # Ensure dy is valid if provided (no zeros/negatives/nans)
        dy = None
        if flux_err is not None:
            dy = np.asarray(flux_err, dtype=float)
            bad = (~np.isfinite(dy)) | (dy <= 0)
            if np.any(bad):
                good = dy[~bad]
                if good.size > 0:
                    dy[bad] = np.nanmedian(good)
                else:
                    dy = None

        # GLS by default when fit_mean=True (default); weighted if dy is provided
        ls = LombScargle(time, flux, dy) if dy is not None else LombScargle(time, flux)

        # χ²-based GLS periodogram; standard normalization:
        # P(f) = (chi2_ref - chi2(f)) / chi2_ref  in [0, 1]
        power = ls.power(freq, method="chi2", normalization="standard")

    except Exception as e:
        print("GLS Lomb-Scargle failed:", e)
        power = np.zeros_like(freq)

    if not np.any(np.isfinite(power)):
        print("Warning: LS power is all NaN; falling back to flat zeros")
        power = np.zeros_like(freq)

    print(f"Power stats: min={power.min():.3e}, max={power.max():.3e}, median={np.median(power):.3e}")

    metrics["power_min"] = float(np.nanmin(power))
    metrics["power_max"] = float(np.nanmax(power))
    metrics["power_median"] = float(np.nanmedian(power))
    metrics["threshold"] = float(np.median(power) * 1.2)

    peaks, _ = find_peaks(power, height=np.median(power) * 1.2)
    note = ""

    if len(peaks) == 0:
        print(f"No significant peaks found for {label}")
        best_idx = int(np.argmax(power))
        note = "(fallback: max power)"
    else:
        best_idx = int(peaks[np.argmax(power[peaks])])
        note = "(best significant peak)"

    metrics["n_peaks"] = int(len(peaks))
    metrics["note"] = str(note)

    best_freq = freq[best_idx]
    best_period = 1 / best_freq

    # --- Peak height metrics (absolute + relative) ---
    best_power = float(power[best_idx])
    metrics["best_power"] = best_power

    power_med = float(np.nanmedian(power))
    peak_ratio = (best_power / power_med) if np.isfinite(power_med) and power_med > 0 else np.nan

    metrics["best_power_over_median"] = peak_ratio

    # --- Flat floor + non-flat region detection (for plotting focus) ---
    floor_power, floor_lp, floor_mad_lp, f_nonflat_max, lp_smooth, thr_lp = \
        estimate_flat_floor_and_nonflat_max(freq, power, hi_frac=0.20, smooth_frac=0.02, k=3.0)

    metrics["power_floor_flat"] = float(floor_power) if np.isfinite(floor_power) else np.nan
    metrics["floor_log10"] = float(floor_lp) if np.isfinite(floor_lp) else np.nan
    metrics["floor_mad_log10"] = float(floor_mad_lp) if np.isfinite(floor_mad_lp) else np.nan
    metrics["f_nonflat_max"] = float(f_nonflat_max) if np.isfinite(f_nonflat_max) else np.nan

    if np.isfinite(floor_power) and floor_power > 0:
        metrics["best_power_over_flat_floor"] = float(best_power / floor_power)
    else:
        metrics["best_power_over_flat_floor"] = np.nan

    # --- Local floor near the best peak (often more meaningful than global flat floor) ---
    local_floor = local_periodogram_floor(power, best_idx, half_width=3000, exclude=200)
    metrics["power_floor_local"] = float(local_floor) if np.isfinite(local_floor) else np.nan
    if np.isfinite(local_floor) and local_floor > 0:
        metrics["best_power_over_local_floor"] = float(best_power / local_floor)
    else:
        metrics["best_power_over_local_floor"] = np.nan


    metrics["best_idx"] = int(best_idx)
    metrics["best_freq"] = float(best_freq)
    metrics["best_period_days"] = float(best_period)

    # FWHM-based uncertainty
    half_max = power[best_idx] / 2.0
    left_idxs = np.where(power[:best_idx] < half_max)[0]
    left_idx = int(left_idxs[-1]) if len(left_idxs) > 0 else 0
    right_idxs = np.where(power[best_idx:] < half_max)[0]
    right_idx = int(best_idx + right_idxs[0]) if len(right_idxs) > 0 else (len(power) - 1)

    fwhm = freq[right_idx] - freq[left_idx]
    period_err = fwhm / (best_freq ** 2)
    print(f"Best period = {best_period:.5f} ± {period_err:.5f} days (from FWHM)")

    metrics["period_err_days"] = float(period_err)

    # reject periods longer than baseline
    if best_period > T:
        print(f"Rejecting unrealistic best_period={best_period:.2f} > baseline={T:.2f}")
        sorted_idx = np.argsort(power)[::-1]
        for idx in sorted_idx:
            candidate_period = 1 / freq[idx]
            if candidate_period <= T:
                best_idx = int(idx)
                best_freq = freq[best_idx]
                best_period = candidate_period
                note = "(corrected: within baseline)"
                break

    periods = [best_period, 2 * best_period]
    T0 = np.floor(np.min(time))

    metrics["T0_btjd"] = float(T0)

    # global y-limits for comparability
    low_global, high_global = np.percentile(flux, [0.1, 99])
    ymin_global = low_global - 0.1 * (high_global - low_global)
    ymax_global = high_global + 0.1 * (high_global - low_global)

    # -----------------------------
    # Save global periodogram (freq, power) to the star folder
    # -----------------------------
    pg_df = pd.DataFrame({
        "frequency_1_per_day": freq.astype(float),
        "period_days": (1.0 / freq).astype(float),
        "power": power.astype(float),
    })

    # gzipped CSV keeps it small but still easy to read later (Python/MATLAB)
    pg_path = os.path.join(star_dir, "periodogram.csv.gz")
    pg_df.to_csv(pg_path, index=False, compression="gzip")

    # Optional: also save as a compact binary for fast reload in Python
    pg_npz_path = os.path.join(star_dir, "periodogram.npz")
    np.savez_compressed(pg_npz_path, freq=freq.astype(float), power=power.astype(float))

    # Save metadata so you don’t need to parse it from the plots
    pg_meta_path = os.path.join(star_dir, "periodogram_meta.json")
    pg_meta = {
        "label": label,
        "target_id": target_id,
        "fmin": float(freq.min()),
        "fmax": float(freq.max()),
        "n_freq": int(len(freq)),
        "baseline_days": float(metrics.get("baseline_days", np.nan)),
        "best_freq": float(metrics.get("best_freq", np.nan)),
        "best_period_days": float(metrics.get("best_period_days", np.nan)),
        "period_err_days": float(metrics.get("period_err_days", np.nan)),
        "best_power": float(metrics.get("best_power", np.nan)),
        "power_median": float(metrics.get("power_median", np.nan)),
        "threshold": float(metrics.get("threshold", np.nan)),
    }
    with open(pg_meta_path, "w") as f:
        json.dump(pg_meta, f, indent=2)

    # record paths in metrics (useful for batch table)
    metrics["periodogram_csv_gz"] = pg_path
    metrics["periodogram_npz"] = pg_npz_path
    metrics["periodogram_meta_json"] = pg_meta_path


    if plot_by_sector:
        # FIGURE 1: periodogram + folded
        fig1, axes1 = plt.subplots(1, 3, figsize=(30, 6))
        fig1.suptitle(fig_title, fontsize=20, y=0.98)

        # --- Periodogram panel (focused on non-flat region + relative height) ---
        ax_pg = axes1[0]

        # Do NOT zoom: always show the full frequency range
        use_rel = False  # keep your behavior (you can turn this on later if you want)
        floor_power = metrics.get("power_floor_flat", np.nan)

        p_plot = (power / floor_power) if (use_rel and np.isfinite(floor_power) and floor_power > 0) else power
        thr = (metrics["threshold"] / floor_power) if (use_rel and np.isfinite(floor_power) and floor_power > 0) else \
        metrics["threshold"]

        ax_pg.plot(freq, p_plot, "-", lw=1)

        ax_pg.axvline(
            best_freq, ls="--", lw=2,
            label=(f"Best: P={best_period:.3f} d | peak={metrics.get('best_power_over_flat_floor', np.nan):.2f}×floor"
                   if (use_rel and np.isfinite(floor_power) and floor_power > 0)
                   else f"Best: P={best_period:.3f} d")
        )
        ax_pg.axhline(thr, ls="--", alpha=0.7, label="Threshold")

        # FULL RANGE
        ax_pg.set_xlim(freq.min(), freq.max())

        # Optional: keep gentle y autoscaling (but not tied to a zoomed x-range)
        m_plot = np.isfinite(p_plot)
        if np.sum(m_plot) > 100:
            ylo, yhi = np.nanpercentile(p_plot[m_plot], [1, 99.99])
            if np.isfinite(ylo) and np.isfinite(yhi) and yhi > ylo:
                ax_pg.set_ylim(max(0, 0.9 * ylo), 1.1 * yhi)

        ax_pg.set_title("LS | " + f"peak/median={metrics.get('best_power_over_median', np.nan):.2f}",fontsize=14)
        ax_pg.set_xlabel("Frequency [1/days]",fontsize=12)
        ax_pg.set_ylabel("Power",fontsize=12)
        ax_pg.grid()
        ax_pg.legend(fontsize=12)

        # --- Folded at P_best ---
        ax_fold = axes1[1]
        phase = ((time - T0) / best_period) % 1
        sort_idx = np.argsort(phase)
        phase, folded_flux = phase[sort_idx], flux[sort_idx]
        folded_err = (flux_err[sort_idx] if flux_err is not None else np.ones_like(folded_flux))

        finite_mask = np.isfinite(phase) & np.isfinite(folded_flux) & np.isfinite(folded_err)
        phase, folded_flux, folded_err = phase[finite_mask], folded_flux[finite_mask], folded_err[finite_mask]

        if len(phase) >= 20:
            fit_par, fit_par_err, y_fit = fitfunction_all_points(folded_flux, folded_err, phase, N=3)
            ms_fold = marker_size_for_panel(len(phase), n_thresh=2000)
            ax_fold.plot(phase, folded_flux, ".", alpha=0.3, ms=ms_fold, label="Data")
            ax_fold.set_title(
                rf"Folded: $P={best_period:.3f}\pm{period_err:.3f}\,\mathrm{{d}},\ T_0={T0:.3f}\,\mathrm{{BTJD}}$",fontsize=14
            )
            ax_fold.set_xlabel("Phase",fontsize=12)
            ax_fold.set_ylabel("Norm. Flux",fontsize=12)
            ax_fold.grid()
            #ax_fold.legend(loc="lower left", fontsize=11)
            ax_fold.set_ylim(ymin_global, ymax_global)
        else:
            ax_fold.text(0.5, 0.5, "Too few points", ha="center", va="center")


        # --- Folded at 2*P_best ---
        ax_fold2 = axes1[2]
        p2 = 2.0 * best_period
        phase2 = ((time - T0) / p2) % 1
        sort_idx2 = np.argsort(phase2)
        phase2, folded_flux2 = phase2[sort_idx2], flux[sort_idx2]
        folded_err2 = (flux_err[sort_idx2] if flux_err is not None else np.ones_like(folded_flux2))

        finite_mask2 = np.isfinite(phase2) & np.isfinite(folded_flux2) & np.isfinite(folded_err2)
        phase2, folded_flux2, folded_err2 = phase2[finite_mask2], folded_flux2[finite_mask2], folded_err2[finite_mask2]

        if len(phase2) >= 20:
            fit_par2, fit_par_err2, y_fit2 = fitfunction_all_points(folded_flux2, folded_err2, phase2, N=3)
            ms_fold = marker_size_for_panel(len(phase2), n_thresh=2000)
            ax_fold2.plot(phase2, folded_flux2, ".", alpha=0.3, ms=ms_fold, label="Data")
            ax_fold2.set_title(rf"Folded: $P=2P_{{\rm best}}={p2:.3f}\,\mathrm{{d}}$",fontsize=14)
            ax_fold2.set_xlabel("Phase",fontsize=12)
            ax_fold2.set_ylabel("Norm. Flux",fontsize=12)
            ax_fold2.grid()
            #ax_fold2.legend(loc="lower left", fontsize=11)
            ax_fold2.set_ylim(ymin_global, ymax_global)
        else:
            ax_fold2.text(0.5, 0.5, "Too few points", ha="center", va="center")

        plt.tight_layout()
        fig1.savefig(fig1_path, dpi=200)
        plt.close(fig1)

        # -----------------------------
        # FIGURE 2: per-sector row layout
        #   per_sector_periodograms=True  -> 4 cols: raw | LS | folded(Pbest_s) | folded(2*Pbest_s)
        #   per_sector_periodograms=False -> keep your old 3 cols: raw | folded(global best) | folded(2*global best)
        # -----------------------------

        def _plot_fold(ax, t_sec, f_sec, e_sec, q_sec, period, T0, title_suffix):
            phase_sec = ((t_sec - T0) / period) % 1
            sidx = np.argsort(phase_sec)

            phase_sec = phase_sec[sidx]
            f_sorted = f_sec[sidx]
            e_sorted = e_sec[sidx] if e_sec is not None else np.ones_like(f_sorted)
            q_sorted = q_sec[sidx]

            good = (q_sorted == 0)
            flag = ~good

            phase_g = phase_sec[good]
            f_g = f_sorted[good]

            ms_g = marker_size_for_panel(len(phase_g), n_thresh=2000)
            ax.plot(
                phase_g, f_g, ".", color="k", alpha=0.3, ms=ms_g,
                label=(f"Good (Q=0), N={len(phase_g)}" if show_flagged_points else None)
            )

            if show_flagged_points and np.any(flag):
                phase_b = phase_sec[flag]
                f_b = f_sorted[flag]
                ms_b = marker_size_for_panel(len(phase_b), n_thresh=2000)
                flag_label = quality_summary(
                    q_sorted[flag], frac_thresh=0.10, max_flags=2,
                    denom="flagged", prefix="Flag"
                ) or "Flagged"
                ax.plot(
                    phase_b, f_b, "x", color="tab:red", alpha=0.3, ms=ms_b,
                    label=f"{flag_label}, N={len(phase_b)}"
                )

            ax.set_title(title_suffix, fontsize=14)
            ax.set_xlabel("Phase", fontsize=12)
            ax.set_ylabel("Norm. Flux", fontsize=12)
            ax.set_ylim(ymin_global, ymax_global)
            ax.grid()

            if show_flagged_points and np.any(flag):
                ax.legend(fontsize=12, loc="best")

        if per_sector_periodograms:
            # ---- 1) precompute per-sector LS so we can enforce identical axis limits across sectors
            sector_sms = []
            sector_powers = []
            for (t_sec, f_sec, e_sec, q_sec, mi) in zip(times, fluxes, fluxerrs, quals, meta):
                good = (q_sec == 0) & np.isfinite(t_sec) & np.isfinite(f_sec)
                tg = t_sec[good]
                fg = f_sec[good]
                eg = e_sec[good] if (e_sec is not None and np.any(e_sec)) else None

                sm = compute_gls_metrics(tg, fg, eg, freq)
                sector_sms.append(sm)
                sector_powers.append(sm["power"])

            # Save sector periodograms for cache reuse (no strings, only numeric arrays)
            sector_npz_path = os.path.join(star_dir, "sector_periodograms.npz")
            sectors_arr = np.array([
                (meta[i].get("sector") if meta[i].get("sector") is not None else -1)
                for i in range(n_sectors)
            ], dtype=np.int32)

            pow2d = np.vstack([np.asarray(sm["power"], float) for sm in sector_sms]).astype(np.float32)
            best_freq_arr = np.array([sm["best_freq"] for sm in sector_sms], dtype=np.float32)
            best_per_arr  = np.array([sm["best_period_days"] for sm in sector_sms], dtype=np.float32)
            perr_arr      = np.array([sm["period_err_days"] for sm in sector_sms], dtype=np.float32)
            thr_arr       = np.array([sm["threshold"] for sm in sector_sms], dtype=np.float32)
            pom_arr       = np.array([sm.get("best_power_over_median", np.nan) for sm in sector_sms], dtype=np.float32)

            np.savez_compressed(
                sector_npz_path,
                freq=freq.astype(np.float32),
                sectors=sectors_arr,
                power=pow2d,
                best_freq=best_freq_arr,
                best_period=best_per_arr,
                period_err=perr_arr,
                threshold=thr_arr,
                peak_over_median=pom_arr,
            )
            metrics["sector_periodograms_npz"] = sector_npz_path

            # common LS x/y limits
            xlim_pg = (float(freq.min()), float(freq.max()))

            allp = np.concatenate([p[np.isfinite(p)] for p in sector_powers if p is not None])
            if allp.size == 0:
                y_max_pg = 1.0
            else:
                # robust "common" ymax so all panels share the same range
                y_max_pg = float(np.nanpercentile(allp, 99.9))
                y_max_pg = max(y_max_pg, float(np.nanmax(allp)) * 0.2)  # safety
                if not np.isfinite(y_max_pg) or y_max_pg <= 0:
                    y_max_pg = float(np.nanmax(allp)) if np.isfinite(np.nanmax(allp)) else 1.0

            ylim_pg = (0.0, 1.05 * y_max_pg)

            # ---- 2) one row per sector, 4 columns
            fig_height = max(4.0, n_sectors * 3.2)
            fig2, axes2 = plt.subplots(
                nrows=n_sectors, ncols=4,
                figsize=(26, fig_height),
                gridspec_kw={"width_ratios": [1.35, 1.0, 1.0, 1.0]}
            )
            if n_sectors == 1:
                axes2 = np.array([axes2])
            axes2 = np.atleast_2d(axes2)

            suptithight = 1 - 0.05 / max(n_sectors, 1)
            fig2.suptitle(fig_title, fontsize=20, y=suptithight)

            for i, (t_sec, f_sec, e_sec, q_sec) in enumerate(zip(times, fluxes, fluxerrs, quals)):
                mi = meta[i]
                sm = sector_sms[i]

                sec_str = f"S{mi['sector']}" if mi.get("sector", None) is not None else f"Idx{i + 1}"
                auth_str = mi.get("author", "unknown") or "unknown"
                flux_kind_str = mi.get("flux_kind", "UNKNOWN")

                nraw = int(mi.get("n_raw", len(t_sec)))
                nkept = int(len(t_sec))
                n_out = int(mi.get("n_outliers_removed", max(nraw - nkept, 0)))
                ngood = int(mi.get("n_kept_good", int(np.sum(q_sec == 0))))
                nflag = int(mi.get("n_kept_flagged", int(np.sum(q_sec != 0))))

                # Column 1: raw
                ax_raw = axes2[i, 0]
                good = (q_sec == 0)
                flag = ~good
                ms_good = marker_size_for_panel(int(np.sum(good)), n_thresh=2000)
                ms_flag = marker_size_for_panel(int(np.sum(flag)), n_thresh=2000)

                ax_raw.plot(t_sec[good], f_sec[good], ".", color="k", markersize=ms_good, alpha=0.7,
                            label=f"Good (Q=0): {ngood}/{nraw}" if show_flagged_points else None)

                if show_flagged_points and np.any(flag):
                    flag_label = quality_summary(
                        q_sec[flag], frac_thresh=0.10, max_flags=2,
                        denom="flagged", prefix="Flag"
                    ) or "Flagged"
                    ax_raw.plot(
                        t_sec[flag], f_sec[flag], "x", color="tab:red",
                        markersize=ms_flag, alpha=0.7,
                        label=f"{flag_label}: {nflag}/{nraw}"
                    )
                soft_line = mi.get("quality_soft_summary_raw", "")

                if show_flagged_points:
                    title = f"{sec_str} | {auth_str} | {flux_kind_str} | Good={ngood}/{nraw} (Flagged={nflag})"
                    if soft_line:
                        title += "\n" + soft_line
                else:
                    title = f"{sec_str} | {auth_str} | {flux_kind_str} | Good={ngood}/{nraw}"

                ax_raw.set_title(title,fontsize=14)
                ax_raw.set_xlabel("Time [days]",fontsize=12)
                ax_raw.set_ylabel("Norm. Flux",fontsize=12)
                ax_raw.set_ylim(ymin_global, ymax_global)
                ax_raw.grid()

                if show_flagged_points and np.any(flag):
                    ax_raw.legend(fontsize=12, loc="best")

                # Column 2: LS (sector best)
                ax_pg = axes2[i, 1]
                power_s = sm["power"]
                fbest_s = sm["best_freq"]
                pbest_s = sm["best_period_days"]
                perr_s = sm["period_err_days"]

                ax_pg.plot(freq, power_s, "-", lw=1)
                ax_pg.axvline(fbest_s, ls="--", lw=2, label=f"Pbest={pbest_s:.3f} d")
                ax_pg.axhline(sm["threshold"], ls="--", alpha=0.7, label="Threshold")
                ax_pg.set_xlim(*xlim_pg)
                ax_pg.set_ylim(*ylim_pg)
                ax_pg.set_title(f"{sec_str} LS | peak/median={sm.get('best_power_over_median', np.nan):.2f}",fontsize=14)
                ax_pg.set_xlabel("Frequency [1/days]",fontsize=12)
                ax_pg.set_ylabel("Power",fontsize=12)
                ax_pg.grid()
                ax_pg.legend(fontsize=12, loc="best")

                # Column 3: folded at sector Pbest
                ax_f1 = axes2[i, 2]
                _plot_fold(
                    ax_f1, t_sec, f_sec, e_sec, q_sec,
                    pbest_s, T0,
                    rf"{sec_str} Folded: $P={pbest_s:.3f}\pm{perr_s:.3f}\,\mathrm{{d}}$"
                )

                # Column 4: folded at 2*sector Pbest
                ax_f2 = axes2[i, 3]
                p2s = 2.0 * pbest_s
                _plot_fold(
                    ax_f2, t_sec, f_sec, e_sec, q_sec,
                    p2s, T0,
                    rf"{sec_str} Folded: $P=2P_{{\rm best}}={p2s:.3f}\,\mathrm{{d}}$"
                )

            plt.tight_layout()
            fig2.savefig(fig2_path, dpi=200)
            plt.close(fig2)

        else:
            # ---- keep your existing 3-column per-sector raw + folded(global) + folded(2*global)
            fig_height = n_sectors * 4
            fig2, axes2 = plt.subplots(nrows=n_sectors, ncols=3, figsize=(18, fig_height),
                                       gridspec_kw={"width_ratios": [1, 1, 1]})
            if n_sectors == 1:
                axes2 = np.array([axes2])
            axes2 = np.atleast_2d(axes2)

            suptithight = 1 - 0.05 / n_sectors
            fig2.suptitle(fig_title, fontsize=20, y=suptithight)

            p2 = 2.0 * best_period

            for i, (t_sec, f_sec, e_sec, q_sec) in enumerate(zip(times, fluxes, fluxerrs, quals)):
                mi = meta[i]
                sec_str = f"S{mi['sector']}" if mi.get("sector", None) is not None else f"Idx{i + 1}"
                auth_str = mi.get("author", "unknown") or "unknown"
                nraw = int(mi.get("n_raw", len(t_sec)))
                n_out = int(mi.get("n_outliers_removed", max(nraw - len(t_sec), 0)))
                ngood = int(mi.get("n_kept_good", int(np.sum(q_sec == 0))))
                nflag = int(mi.get("n_kept_flagged", int(np.sum(q_sec != 0))))

                ax_left = axes2[i, 0]
                good = (q_sec == 0)
                flag = ~good

                ms_good = marker_size_for_panel(int(np.sum(good)), n_thresh=2000)
                ms_flag = marker_size_for_panel(int(np.sum(flag)), n_thresh=2000)

                ax_left.plot(
                    t_sec[good], f_sec[good], ".", color="k", markersize=ms_good, alpha=0.7,
                    label=(f"Good (Q=0): {ngood}/{nraw}" if show_flagged_points else None)
                )

                if show_flagged_points and np.any(flag):
                    flag_label = quality_summary(
                        q_sec[flag], frac_thresh=0.10, max_flags=2,
                        denom="flagged", prefix="Flag"
                    ) or "Flagged"
                    ax_left.plot(
                        t_sec[flag], f_sec[flag], "x", color="tab:red",
                        markersize=ms_flag, alpha=0.7,
                        label=f"{flag_label}: {nflag}/{nraw} | outliers={n_out}"
                    )

                soft_line = mi.get("quality_soft_summary_raw", "")

                if show_flagged_points:
                    title = f"{sec_str} | {auth_str} | Good={ngood}/{nraw} (Flagged={nflag})"
                    if soft_line:
                        title += "\n" + soft_line
                else:
                    title = f"{sec_str} | {auth_str} | Good={ngood}/{nraw}"

                ax_left.set_title(title)

                if show_flagged_points and np.any(flag):
                    ax_left.legend(fontsize=10, loc="best")
                ax_left.set_xlabel("Time [days]")
                ax_left.set_ylabel("Norm. Flux")
                ax_left.set_ylim(ymin_global, ymax_global)
                ax_left.grid()

                ax_p = axes2[i, 1]
                ax_2p = axes2[i, 2]

                _plot_fold(ax_p, t_sec, f_sec, e_sec, q_sec, best_period, T0,
                           rf"{sec_str} Folded: $P={best_period:.3f}\,\mathrm{{d}}$")
                _plot_fold(ax_2p, t_sec, f_sec, e_sec, q_sec, p2, T0,
                           rf"{sec_str} Folded: $P=2P_{{\rm best}}={p2:.3f}\,\mathrm{{d}}$")

            plt.tight_layout()
            fig2.savefig(fig2_path, dpi=200)
            plt.close(fig2)


    else:
        # Default mode: one figure (no per-sector panels)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ms_raw = marker_size_for_panel(len(time), n_thresh=2000)
        axes[0, 0].plot(time, flux, ".", markersize=ms_raw, alpha=0.7)
        axes[0, 0].set_title(f"{label} Lightcurve")
        axes[0, 0].set_xlabel("Time [days]")
        axes[0, 0].set_ylabel("Normalized Flux")
        axes[0, 0].grid()

        axes[0, 1].axvline(best_freq, color="m", ls="--", lw=2, label=f"Best Period: {best_period:.3f} d")
        axes[0, 1].plot(freq, power, "k-", lw=1)
        threshold = np.median(power) * 1.2
        axes[0, 1].axhline(threshold, color="gray", ls="--", alpha=0.7, label="Threshold")
        axes[0, 1].set_xlim(freq.min(), freq.max())
        axes[0, 1].set_xlabel("Frequency [1/days]")
        axes[0, 1].set_ylabel("Power")
        axes[0, 1].set_title("Lomb–Scargle Periodogram")
        axes[0, 1].grid()
        axes[0, 1].legend(fontsize=15)

        # Folded curves
        for i, p in enumerate(periods):
            phase = ((time - T0) / p) % 1
            sort_idx = np.argsort(phase)
            phase = phase[sort_idx]
            folded_flux = flux[sort_idx]
            folded_err = (flux_err[sort_idx] if flux_err is not None else np.ones_like(folded_flux))

            finite_mask = np.isfinite(phase) & np.isfinite(folded_flux) & np.isfinite(folded_err)
            phase, folded_flux, folded_err = phase[finite_mask], folded_flux[finite_mask], folded_err[finite_mask]

            ax = axes[1, i]
            if len(phase) < 20:
                ax.text(0.5, 0.5, "Too few points", ha="center", va="center")
                ax.set_title(f"Folded: P={p:.3f} d")
                continue

            fit_par, fit_par_err, y_fit = fitfunction_all_points(folded_flux, folded_err, phase, N=3)
            fit_text = make_fit_quadrature_string_N(fit_par, fit_par_err, N=3)

            ms_fold = marker_size_for_panel(len(phase), n_thresh=2000)
            ax.plot(phase, folded_flux, ".", alpha=0.3, ms=ms_fold, label="Data")
            ax.set_title(f"Folded: P={p:.3f} d {note if i == 0 else ''}")
            ax.set_xlabel("Phase")
            ax.set_ylabel("Norm. Flux")
            ax.set_ylim(ymin_global, ymax_global)
            ax.grid()
            ax.legend(loc="lower left", fontsize=15)
            ax.text(0.5, 0.97, fit_text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=11,
                    bbox=dict(boxstyle="round", alpha=0.2))

        plt.tight_layout()
        fname = "per_sector_raw_periodogram_and_folded" if per_sector_periodograms else "per_sector_raw_and_folded"
        fig.savefig(os.path.join(star_dir, fname + ".png"), dpi=200)
        #fig2.savefig(os.path.join(outdir, fname + ".pdf"), dpi=200)
        plt.close(fig)

    # Save cleaned raw lightcurve
    sector_col = np.concatenate([
        np.full(len(t), (m.get("sector") if m.get("sector") is not None else -1), dtype=int)
        for t, m in zip(times, meta)
    ])
    author_col = np.concatenate([
        np.full(len(t), (m.get("author") or "unknown"), dtype=object)
        for t, m in zip(times, meta)
    ])
    fluxkind_col = np.concatenate([
        np.full(len(t), (m.get("flux_kind") or "UNKNOWN"), dtype=object)
        for t, m in zip(times, meta)
    ])


    lc_df = pd.DataFrame({
        "sector": sector_col,
        "author": author_col,
        "flux_kind": fluxkind_col,
        "time_btjd": time_all,
        "flux_norm": flux_all,
        "flux_err": flux_err_all,
        "quality": qual_all,
        "is_good": (qual_all == 0).astype(int),
    })

    lc_path = os.path.join(star_dir, "cleaned_lightcurve.csv")
    lc_df.to_csv(lc_path, index=False)
    print(f"Saved cleaned lightcurve to {lc_path}")

    metrics["good_points_per_sector"] = json.dumps([m.get("n_kept_good", None) for m in meta])
    metrics["flagged_points_per_sector"] = json.dumps([m.get("n_kept_flagged", None) for m in meta])
    metrics["quality_bits_present_per_sector"] = json.dumps([m.get("quality_bits_present_raw", []) for m in meta])
    metrics["quality_names_present_per_sector"] = json.dumps([m.get("quality_names_present_raw", []) for m in meta])
    metrics["quality_soft_summary_per_sector"] = json.dumps([m.get("quality_soft_summary_raw", "") for m in meta])

    metrics["outdir"] = star_dir
    metrics["cleaned_lightcurve_csv"] = lc_path

    print(f"Done {label}, best period={best_period:.3f} d {note}")
    return _finish("ok", "")


# -----------------------------
# Batch runner from CSV
# -----------------------------
def run_batch_from_csv(
    csv_path,
    output_dir="TESSPlots",
    search_radius=2*u.arcmin,
    tic_col=None, gaia_col=None, ra_col=None, dec_col=None,
    interactive=False,
    plot_by_sector_default=True,
    fmin=None, fmax=None,
    output_csv_path=None,
    flux_mode="default",
    per_sector_periodograms=False,
    plot_output_layout="by_star",
    cache_policy="auto",
    show_flagged_points=False,
):
    df = pd.read_csv(csv_path, dtype=str)

    cols_lower = {c.strip().lower(): c for c in df.columns}

    per_col   = cols_lower.get("per", None)
    psini_col = cols_lower.get("psini", None)
    period_col = cols_lower.get("gaia period", None) or cols_lower.get("gaia_period", None)
    gmag_col = cols_lower.get("gmag", None) or cols_lower.get("g_mag", None) or cols_lower.get("phot_g_mean_mag", None)

    if tic_col is None and gaia_col is None and ra_col is None and dec_col is None:
        tic_col, gaia_col, ra_col, dec_col = infer_columns(df)

    print("CSV columns detected/used:")
    print(f"  TIC column : {tic_col}")
    print(f"  Gaia column: {gaia_col}")
    print(f"  RA column  : {ra_col}")
    print(f"  Dec column : {dec_col}")

    metrics_rows = []
    n_total = len(df)
    n_ok = 0
    n_fail = 0

    # --- progress bar over stars ---
    iterable = df.iterrows()
    if tqdm is not None:
        pbar = tqdm(iterable, total=len(df), desc="Batch stars", unit="star", dynamic_ncols=True)
    else:
        pbar = iterable

    for idx, row in pbar:
        tic_val = None
        gaia_val = None
        ra_val = None
        dec_val = None

        # Prefer TIC
        if tic_col is not None:
            v = row.get(tic_col, None)
            if v is not None and str(v).strip() != "" and str(v).lower() != "nan":
                tic_val = str(v).split(".")[0].strip()

        # Else Gaia source_id
        if tic_val is None and gaia_col is not None:
            v = row.get(gaia_col, None)
            if v is not None and str(v).strip() != "" and str(v).lower() != "nan":
                gaia_val = str(v).strip()  # do NOT split '.'; parser handles sci/decimals safely

        # Else coordinates
        if tic_val is None and gaia_val is None:
            if ra_col is not None and dec_col is not None:
                ra_s = row.get(ra_col, "")
                dec_s = row.get(dec_col, "")
                try:
                    ra_val = float(ra_s)
                    dec_val = float(dec_s)
                    if not (np.isfinite(ra_val) and np.isfinite(dec_val)):
                        raise ValueError("non-finite ra/dec")
                except Exception:
                    # record failed row
                    metrics_rows.append({
                        "status": "missing_target_info",
                        "error": "No valid TIC/Gaia/ra/dec in row",
                        "label": "",
                        "target_id": "",
                        "tic_id": str(tic_val) if tic_val else "",
                        "gaia_source_id": str(gaia_val) if gaia_val else "",
                        "ra_deg": np.nan,
                        "dec_deg": np.nan,
                    })
                    n_fail += 1
                    continue
            else:
                metrics_rows.append({
                    "status": "missing_target_info",
                    "error": "No TIC/Gaia and no ra/dec columns",
                    "label": "",
                    "target_id": "",
                    "tic_id": str(tic_val) if tic_val else "",
                    "gaia_source_id": str(gaia_val) if gaia_val else "",
                    "ra_deg": np.nan,
                    "dec_deg": np.nan,
                })
                n_fail += 1
                continue

        try:
            per_val = row.get(per_col, None) if per_col is not None else None
            psini_val = row.get(psini_col, None) if psini_col is not None else None
            gaia_period_val = row.get(period_col, None) if period_col is not None else None
            gmag_val = row.get(gmag_col, None) if gmag_col is not None else None

            m = analyze_tess_lightcurve(
                tic_id=tic_val,
                gaia_source_id=gaia_val,
                ra=ra_val, dec=dec_val,
                output_dir=output_dir,
                search_radius=search_radius,
                interactive=interactive,
                plot_by_sector_default=plot_by_sector_default,
                fmin=fmin, fmax=fmax,
                return_metrics=True,
                batch_row_index=idx,
                per=per_val,
                psini=psini_val,
                gaia_period=gaia_period_val,
                flux_mode=flux_mode,
                gmag=gmag_val,
                per_sector_periodograms=per_sector_periodograms,
                plot_output_layout=plot_output_layout,
                cache_policy=cache_policy,
                show_flagged_points=show_flagged_points,
            )
            metrics_rows.append(m if m is not None else {"status": "unknown", "error": "No metrics returned"})
            status = (m or {}).get("status", "")
            if status in ("ok", "ok_cached"):
                n_ok += 1
            else:
                n_fail += 1

            if tqdm is not None and hasattr(pbar, "set_postfix"):
                pbar.set_postfix(ok=n_ok, failed=n_fail)

        except Exception as e:
            metrics_rows.append({"status": "exception", "error": str(e)})
            n_fail += 1

    metrics_df = pd.DataFrame(metrics_rows)

    # Align rows: one metrics row per input row. If anything went off, force length match.
    if len(metrics_df) != len(df):
        # pad/truncate to match
        metrics_df = metrics_df.reindex(range(len(df)))

    out_df = pd.concat([df.reset_index(drop=True), metrics_df.reset_index(drop=True)], axis=1)

    if output_csv_path is None:
        base, ext = os.path.splitext(csv_path)
        output_csv_path = base + "_with_TESS_metrics.csv"

    out_df.to_csv(output_csv_path, index=False)
    print(f"Batch complete: ok={n_ok}, failed={n_fail}, total={n_total}")
    print(f"Saved augmented table to: {output_csv_path}")



# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    flux_mode = input("Flux mode: [d]etrended, [r]aw(SAP), [D]efault(SPOC=PDCSAP,QLP=SAP): ").strip().lower()
    if flux_mode in ["d", "det", "detrended"]:
        flux_mode = "detrended"
    elif flux_mode in ["r", "raw", "sap"]:
        flux_mode = "raw"
    else:
        flux_mode = "default"

    per_sec = input("Per-sector periodograms (extra row + per-sector best P)? [y/N]: ").strip().lower()
    per_sector_periodograms = per_sec in ["y", "yes"]

    cache_in = input("Cache policy: [a]uto(use if exists), [u]se-only, [r]efresh(recompute)? [a]: ").strip().lower()
    if cache_in.startswith("u"):
        cache_policy = "use"
    elif cache_in.startswith("r"):
        cache_policy = "refresh"
    else:
        cache_policy = "auto"

    mode = input("Input mode: [1] TIC, [2] Gaia source_id, [3] coordinates, [4] CSV batch? (1/2/3/4): ").strip()

    if mode == "1":
        tic_id = input("Enter TIC ID: ").strip()
        analyze_tess_lightcurve(tic_id=tic_id,
                                flux_mode=flux_mode,
                                per_sector_periodograms=per_sector_periodograms,
                                cache_policy=cache_policy)

    elif mode == "2":
        gaia_sid = input("Enter Gaia DR3 source_id: ").strip()
        analyze_tess_lightcurve(tic_id=None,
                                gaia_source_id=gaia_sid,
                                ra=None, dec=None,
                                flux_mode=flux_mode,
                                per_sector_periodograms=per_sector_periodograms,
                                cache_policy=cache_policy)

    elif mode == "3":
        ra = float(input("Enter RA [deg]: ").strip())
        dec = float(input("Enter Dec [deg]: ").strip())
        analyze_tess_lightcurve(tic_id=None,
                                gaia_source_id=None,
                                ra=ra, dec=dec,
                                flux_mode=flux_mode,
                                per_sector_periodograms=per_sector_periodograms,
                                cache_policy=cache_policy)

    else:
        if ".csv" in mode:
            csv_path = mode
        else:
            csv_path = input("Enter CSV path: ").strip()

        layout_in = input("Plot output layout: [s]tar folders (default) or [t]ype folders? ").strip().lower()
        plot_output_layout = "by_plot_type" if layout_in.startswith("t") else "by_star"

        run_batch_from_csv(csv_path,
                           output_dir="TESSPlots",
                           interactive=False,
                           flux_mode=flux_mode,
                           per_sector_periodograms=per_sector_periodograms,
                           plot_output_layout=plot_output_layout,
                           cache_policy=cache_policy)

