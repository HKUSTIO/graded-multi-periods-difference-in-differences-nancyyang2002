from __future__ import annotations

import math

import numpy as np
import pandas as pd


def logistic(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(x))))


def generate_panel_data(config: dict, heterogeneous_trend: bool) -> pd.DataFrame:
    rng = np.random.default_rng(int(config["seed_population"]))
    n_units = int(config["n_units"])
    t0 = int(config["pre_periods"])
    t_total = t0 + int(config["post_periods"])
    times = np.arange(1, t_total + 1)

    x_draw = rng.uniform(0.0, 1.0, size=n_units)
    x1 = (x_draw >= 0.3).astype(int)
    x2 = (x_draw >= 0.7).astype(int)
    u = rng.uniform(0.0, 1.0, size=n_units)

    alpha0 = float(config["adoption_intercept"])
    alpha1 = float(config["adoption_slope_middle"])
    alpha2 = float(config["adoption_slope_late"])
    p1 = logistic(alpha0)
    p2 = np.array([logistic(alpha0 + alpha1 * value) for value in x1])
    p3 = np.array([logistic(alpha0 + alpha1 * value) for value in x1 + x2])
    p4 = np.array([logistic(alpha0 + alpha2 * value) for value in x1 + x2])

    cohort = np.zeros(n_units, dtype=int)
    cohort[u <= p1] = t0 + 1
    cohort[(u > p1) & (u <= p2)] = t0 + 2
    cohort[(u > p2) & (u <= p3)] = t0 + 3
    cohort[(u > p3) & (u <= p4)] = t0 + 4

    individual_effect = rng.normal(0.0, float(config["individual_sd"]), size=n_units)
    tau_i = rng.normal(float(config["tau_mean"]), float(config["tau_sd"]), size=n_units)
    time_shocks = rng.uniform(float(config["tau_time_low"]), float(config["tau_time_high"]), size=t_total + 1)

    rows = []
    for time in times:
        if heterogeneous_trend:
            trend_cfg = config["heterogeneous_trend"]
            trend = (
                (time / t_total) * float(trend_cfg["baseline_slope"]) * (1 - x1 - x2)
                + (time / t_total) * float(trend_cfg["x1_slope"]) * x1
                + (time / t_total) * float(trend_cfg["x2_slope"]) * x2
            )
        else:
            trend = np.full(n_units, time / t_total)

        error = rng.normal(0.0, float(config["error_sd"]), size=n_units)
        treated_ever = (cohort > 0).astype(int)
        y0 = float(config["base_level"]) + treated_ever * (-individual_effect) + (1 - treated_ever) * individual_effect + trend + error

        d = ((cohort > 0) & (time >= cohort)).astype(int)
        multiplier = np.zeros(n_units)
        multiplier[cohort == t0 + 1] = 1.0
        multiplier[cohort == t0 + 2] = -2.5
        multiplier[cohort == t0 + 3] = -1.75
        multiplier[cohort == t0 + 4] = -1.0
        tau_it = time_shocks[time] * np.abs(tau_i) * multiplier
        y = y0 + d * tau_it
        relative_time = np.where(cohort > 0, time - cohort, 0)

        for idx in range(n_units):
            rows.append(
                {
                    "id": idx + 1,
                    "x1": int(x1[idx]),
                    "x2": int(x2[idx]),
                    "cohort": int(cohort[idx]),
                    "time": int(time),
                    "relative_time": int(relative_time[idx]),
                    "d": int(d[idx]),
                    "y0": float(y0[idx]),
                    "tau_it": float(tau_it[idx]),
                    "y": float(y[idx]),
                }
            )

    return pd.DataFrame(rows, columns=["id", "x1", "x2", "cohort", "time", "relative_time", "d", "y0", "tau_it", "y"])


def summarize_group_shares_and_att(data: pd.DataFrame) -> pd.DataFrame:
    """
    Return one row per treated cohort and one row for all treated observations.
    """
    n_unique_ids = data["id"].nunique()

    # Cohort stats
    cohort_stats = (
        data.groupby("cohort")
        .apply(
            lambda x: pd.Series(
                {
                    "fraction": x["id"].nunique() / n_unique_ids,
                    "att": x.loc[x["d"] == 1, "tau_it"].mean() if (x["d"] == 1).any() else np.nan,
                }
            )
        )
        .reset_index()
    )

    # Filter for treated cohorts (G > 0)
    treated_cohorts = cohort_stats[cohort_stats["cohort"] > 0].copy()
    treated_cohorts["group"] = treated_cohorts["cohort"].apply(lambda g: f"cohort_{g}")

    # All treated row
    total_obs = len(data)
    treated_obs_mask = data["d"] == 1
    all_treated_fraction = treated_obs_mask.sum() / total_obs
    all_treated_att = data.loc[treated_obs_mask, "tau_it"].mean()

    all_treated_row = pd.DataFrame(
        [{"group": "all_treated", "fraction": all_treated_fraction, "att": all_treated_att}]
    )

    result = pd.concat([treated_cohorts[["group", "fraction", "att"]], all_treated_row], ignore_index=True)
    return result


def estimate_cohort_did(data: pd.DataFrame, cohort: int, event_time: int, control_group: str) -> float:
    """
    Return a two-period DID estimate for one treatment cohort and event time.
    """
    t_target = cohort + event_time
    t_base = cohort - 1

    # Check if times exist in data
    available_times = data["time"].unique()
    if t_target not in available_times or t_base not in available_times:
        return np.nan

    # Treated group: units in cohort 'cohort'
    treated_mask = data["cohort"] == cohort

    # Control group
    if control_group == "never":
        control_mask = data["cohort"] == 0
    elif control_group == "notyet":
        control_mask = (data["cohort"] == 0) | (data["cohort"] > t_target)
    else:
        raise ValueError("control_group must be 'never' or 'notyet'")

    # Helper to get mean outcome for a mask and time
    def get_mean(mask, time):
        subset = data[mask & (data["time"] == time)]
        if subset.empty:
            return np.nan
        return subset["y"].mean()

    y_t_target = get_mean(treated_mask, t_target)
    y_t_base = get_mean(treated_mask, t_base)
    y_c_target = get_mean(control_mask, t_target)
    y_c_base = get_mean(control_mask, t_base)

    if any(np.isnan([y_t_target, y_t_base, y_c_target, y_c_base])):
        return np.nan

    return (y_t_target - y_t_base) - (y_c_target - y_c_base)


def estimate_event_study(data: pd.DataFrame, event_times: list[int], control_group: str) -> pd.DataFrame:
    """
    Return cohort-event DID estimates.
    """
    treated_cohorts = sorted(data[data["cohort"] > 0]["cohort"].unique())
    available_times = set(data["time"].unique())

    results = []
    for g in treated_cohorts:
        for e in event_times:
            t_target = g + e
            t_base = g - 1
            if t_target in available_times and t_base in available_times:
                estimate = estimate_cohort_did(data, g, e, control_group)
                if not np.isnan(estimate):
                    results.append({"cohort": g, "event_time": e, "estimate": estimate})

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return pd.DataFrame(columns=["cohort", "event_time", "estimate"])
    return res_df.sort_values(["cohort", "event_time"]).reset_index(drop=True)


def aggregate_post_treatment_effects(event_study: pd.DataFrame) -> float:
    """
    Return the average estimate over post-treatment event times.
    """
    post_study = event_study[event_study["event_time"] >= 0]
    if post_study.empty:
        return np.nan
    return float(post_study["estimate"].mean())


def estimate_twfe_coefficient(data: pd.DataFrame) -> float:
    """
    Return the coefficient from a residualized two-way fixed effects regression of y on d.
    """
    y_grand_mean = data["y"].mean()
    d_grand_mean = data["d"].mean()

    y_i_mean = data.groupby("id")["y"].transform("mean")
    d_i_mean = data.groupby("id")["d"].transform("mean")

    y_t_mean = data.groupby("time")["y"].transform("mean")
    d_t_mean = data.groupby("time")["d"].transform("mean")

    y_ddot = data["y"] - y_i_mean - y_t_mean + y_grand_mean
    d_ddot = data["d"] - d_i_mean - d_t_mean + d_grand_mean

    denominator = (d_ddot**2).sum()
    if denominator == 0:
        return np.nan
    beta_twfe = (d_ddot * y_ddot).sum() / denominator
    return float(beta_twfe)
