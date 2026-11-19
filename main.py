"""gta6-trophy-rarity-forecast-01 entry (Python 3.13, standard library only).

Forecasts every GTA VI trophy's Exophase earn rate at the case's target as five
quantiles, from the case's history of reads.

Model (per trophy, in logit space, x = ln(1 + hours since 2026-11-19T00:00:00Z)):
  * Distinct reads: consecutive identical reads carry no information, so the fit
    uses only reads where a rate or the players-tracked count changed (plus the
    last read).  A trophy's curve is logit(rate) = a + b*x, fitted by least
    squares with recency weights (24 h half-life), then re-anchored through the
    last observed rate: prediction = logit(last) + b_shrunk * (x_target - x_last).
  * Shrinkage: each trophy's slope b is shrunk toward the median slope of its
    tier (platinum/gold/silver/bronze), with strength set by how many distinct
    reads the trophy has; trophies with almost no movement borrow the tier's
    shape instead of extrapolating noise.
  * Flat history (no refresh ever seen, e.g. the practice cases): the page is
    stale and a refresh is due.  A refresh multiplies the tracked population
    (about 6x on launch night) while the new cohort has earned far fewer
    trophies, so rates fall: the median is shifted down in logit by
    -ln(6) + ln(1 + K * (r / r_max) ** 1.5), i.e. most trophies drop hard and
    the earliest/commonest trophies (highest current rate) drop least, with a
    wide spread to cover how uneven the dilution is across trophies.
  * Spread: fit residual (floored) combined with slope uncertainty, a random
    walk term 0.30 * sqrt(horizon in days), and an ignorance floor; flat
    histories get a much wider spread.  The 100% trophy keeps a small left
    tail (it can only fall as fresh players sync) and never a right tail.

Deterministic: pure arithmetic, no randomness, files processed in name order,
quantiles rounded to 4 decimals half-up.  Reads $MAKEFUN_INPUT_DIR (else
/input), writes $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Release-day origin: hours are measured from 2026-11-19T00:00:00Z.
EPOCH = "2026-11-19T00:00:00Z"

# --- tuned constants (see notes above) -------------------------------------
RW_SIGMA = 0.30        # logit random-walk sd per sqrt(day)
SPREAD_FLOOR = 0.15     # minimum logit sd on any fitted forecast
RESID_FLOOR = 0.15      # minimum per-trophy residual sd
RESID_CAP = 1.0         # maximum per-trophy residual sd
SHRINK_K = 3.0          # prior strength (in distinct reads) for slope shrinkage
SLOPE_MIN = -1.0        # clamp on shrunk slopes
SLOPE_MAX = 3.0         # ... and on the upside. Guard against explosive
# extrapolation over the final's long horizon: x_target - x_last is about
# ln(169) - ln(60.5) = 1.03 there, so a slope of 5 would move the centre
# +5.1 logit (a trophy at 1% forecast at ~62% in 4.5 days -- implausible:
# the player base broadens with slower players, it does not all earn a
# trophy almost nobody had). Slope 3 still allows +3.1 logit (1% -> ~18%),
# generous for a mid-story trophy, but no longer lets a noisy steep fit
# dominate the forecast. Typical fitted/shrunk slopes sit well below 3,
# so this binds only on extreme extrapolations.
FLAT_N_GROWTH = 6.0     # expected tracked-player multiplier at the next refresh
FLAT_GROWTH_K = 1.0
FLAT_SD = 0.75         # logit sd on a flat-history (dilution-shock) forecast.
# Calibration: the one observed refresh multiplied players 6.06x, but the next
# refresh's multiplier is unobserved (plausibly ~3x-12x, +/-0.7 in logit) and
# the new cohort's earn rate per trophy is unobserved too (+/-0.5 in logit),
# so the honest sd is ~0.75, not 0.5: at 0.5 the 90% band spans only a
# 0.44x-2.3x rate ratio around the median, narrower than the plausible
# multiplier range alone. Pinball loss charges 1.9 per logit outside the
# 90% band but only 0.1 per logit inside it, so the wider band is cheap
# insurance against a shock whose size is genuinely unknown.
FLAT_HORIZON_MIN = 720  # flat-history rule only for horizons up to 12 h
TWO_POINT_SHRINK = 0.70  # fraction of slope borrowed from the tier (2 points)
TWO_POINT_SD = 0.55


def parse_time(s):
    """ISO-8601 UTC timestamp -> seconds since the Unix epoch (no dateutil)."""
    # Format: 2026-11-19T03:00:00Z (fractional seconds possible).
    date, time = s.split("T")
    y, mo, d = (int(v) for v in date.split("-"))
    time = time.rstrip("Z")
    if "." in time:
        time, _frac = time.split(".")
    h, mi, sec = (int(v) for v in time.split(":"))
    days = _days_since_epoch(y, mo, d)
    return days * 86400 + h * 3600 + mi * 60 + sec


def _days_since_epoch(y, m, d):
    """Days from 1970-01-01 to y-m-d (Howard Hinnant's algorithm)."""
    y_adj = y - (1 if m <= 2 else 0)
    era = (y_adj if y_adj >= 0 else y_adj - 399) // 400
    yoe = y_adj - era * 400
    mp = (m + 9) % 12
    doy = (153 * mp + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


EPOCH_SEC = parse_time(EPOCH)


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matches the starter's JS rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def distinct_reads(case, idx):
    """Change-point reads for trophy idx: (hours since epoch, rate, players).

    Keeps a read when its rate for this trophy or the players-tracked count
    differs from the previous kept read, plus the final read.  Returns the
    list oldest first, and the last observed rate (skipping nulls).
    """
    pts = []
    last_rate = None
    last_n = None
    for read in case["history"]:
        rate = read["rates"][idx]
        if rate is None:
            continue
        n = read["players_tracked"]
        t = (parse_time(read["at"]) - EPOCH_SEC) / 3600.0
        if t < 0:
            t = 0.0
        if not pts or rate != pts[-1][1] or n != last_n:
            pts.append((t, rate, n))
        last_rate = rate
        last_n = n
    # Always end on the last read (its time matters for anchoring).
    if pts and pts[-1][1] != last_rate:
        t = (parse_time(case["history"][-1]["at"]) - EPOCH_SEC) / 3600.0
        pts.append((max(t, 0.0), last_rate, case["history"][-1]["players_tracked"]))
    return pts, last_rate


def fit_slope(pts):
    """Weighted LSQ slope/intercept of logit(rate) on ln(1+t); recency weights.

    Returns (slope, intercept, resid_sd, x_bar, sxx, n).  Recency weight halves
    every 24 h so the recent (broad-population) regime dominates.
    """
    if len(pts) < 2:
        return None
    t_last = pts[-1][0]
    xs, ys, ws = [], [], []
    for t, rate, _n in pts:
        xs.append(math.log1p(t))
        ys.append(logit(rate))
        ws.append(0.5 ** ((t_last - t) / 24.0))
    sw = sum(ws)
    x_bar = sum(w * x for w, x in zip(ws, xs)) / sw
    y_bar = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (0.0, y_bar, 0.0, x_bar, sxx, len(pts))
    sxy = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    intercept = y_bar - slope * x_bar
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    if len(pts) > 2:
        var = sum(w * r * r for w, r in zip(ws, resid)) / sw
        resid_sd = math.sqrt(var) * math.sqrt(len(pts) / max(1, len(pts) - 2))
    else:
        resid_sd = 0.0
    return (slope, intercept, resid_sd, x_bar, sxx, len(pts))


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("unexpected levels")
    if not case["history"]:
        raise ValueError("no history")
    horizon_days = case["horizon_minutes"] / 1440.0
    target_h = (parse_time(case["target"]) - EPOCH_SEC) / 3600.0
    if target_h < 0:
        target_h = 0.0
    x_target = math.log1p(target_h)
    n_tr = len(case["trophies"])

    # Per-trophy series and raw fits.
    series = []
    fits = []
    for i in range(n_tr):
        pts, last_rate = distinct_reads(case, i)
        series.append((pts, last_rate))
        fits.append(fit_slope(pts) if len(pts) >= 2 else None)

    # Tier-median slopes (from trophies with a real fit) for shrinkage.
    tier_slopes = {}
    for tier in ("platinum", "gold", "silver", "bronze"):
        vals = [f[0] for f, tr in zip(fits, case["trophies"])
                if f is not None and tr["tier"] == tier]
        if vals:
            tier_slopes[tier] = median(vals)
    all_slopes = [f[0] for f in fits if f is not None]
    global_slope = median(all_slopes) if all_slopes else 0.5

    # Highest non-100 rate, for the flat-history dilution shape.
    live_rates = [lr for _pts, lr in series if lr is not None and lr < 99.99]
    r_max = max(live_rates) if live_rates else 1.0

    rw = RW_SIGMA * math.sqrt(max(horizon_days, 0.0))
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        pts, last_rate = series[i]
        if last_rate is None:
            last_rate = 1.0
        tier = trophy["tier"]
        tier_slope = tier_slopes.get(tier, global_slope)

        # The 100% trophy: it can only fall (fresh players syncing).  Keep a
        # small left tail as insurance, never a right tail.
        if last_rate >= 99.99:
            rows.append({"id": trophy["id"], "q": [99.0, 99.9, 100, 100, 100]})
            continue

        if len(pts) <= 1 and case["horizon_minutes"] <= FLAT_HORIZON_MIN:
            # Flat history: no refresh seen yet; one is due inside the horizon.
            # Median shift: dilution -ln(N growth) plus new earns, which
            # concentrate on the commonest trophies.
            frac = min(1.0, case["horizon_minutes"] / 90.0)
            shift = (-math.log(FLAT_N_GROWTH)
                     + math.log1p(FLAT_GROWTH_K * (last_rate / r_max) ** 1.5)) * frac
            centre = logit(last_rate) + shift
            sd = FLAT_SD
        elif len(pts) <= 1:
            # Flat history but a long horizon: no refresh seen, so no basis
            # for a shift; admit wide ignorance instead.
            centre = logit(last_rate)
            sd = math.sqrt(FLAT_SD ** 2 + rw ** 2 + SPREAD_FLOOR ** 2)
        elif len(pts) == 2:
            # One refresh seen: a two-point slope, mostly borrowed from tier.
            f = fits[i]
            slope = f[0] if f else 0.0
            slope = (1.0 - TWO_POINT_SHRINK) * slope + TWO_POINT_SHRINK * tier_slope
            slope = min(SLOPE_MAX, max(SLOPE_MIN, slope))
            x_last = math.log1p(pts[-1][0])
            centre = logit(last_rate) + slope * (x_target - x_last)
            sd = math.sqrt(TWO_POINT_SD ** 2 + rw ** 2 + SPREAD_FLOOR ** 2)
        else:
            f = fits[i]
            slope, _icpt, resid_sd, x_bar, sxx, npts = f
            # Shrink the slope toward the tier median; more reads = less shrink.
            w = npts / (npts + SHRINK_K)
            slope = w * slope + (1.0 - w) * tier_slope
            slope = min(SLOPE_MAX, max(SLOPE_MIN, slope))
            x_last = math.log1p(pts[-1][0])
            centre = logit(last_rate) + slope * (x_target - x_last)
            resid = min(RESID_CAP, max(RESID_FLOOR, resid_sd))
            # Prediction-interval factor for extrapolating the line.
            if sxx > 1e-12:
                lever = 1.0 + (x_target - x_bar) ** 2 / sxx
            else:
                lever = 2.0
            sd = math.sqrt((resid ** 2) * min(lever, 4.0) + rw ** 2 + SPREAD_FLOOR ** 2)

        qs = [round4(min(100.0, max(0.0, inv_logit(centre + z * sd)))) for z in Z]
        # Keep quantiles non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
