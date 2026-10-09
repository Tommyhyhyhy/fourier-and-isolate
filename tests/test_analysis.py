"""Scientific invariants for the chronological FFT / Isolation Forest pipeline."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from weather_lab.analysis import AnalysisConfig, analyze


@pytest.fixture
def weather():
    """Forty hourly days, with known daily temperature and slower wind cycles."""
    rng = np.random.default_rng(20261009)
    t = np.arange(40 * 24, dtype=float)
    daily = np.sin(2 * np.pi * t / 24)
    return pd.DataFrame(
        {
            "date": pd.date_range("2025-09-01", periods=len(t), freq="h", tz="UTC"),
            "temperature_2m": 27 + 5 * daily + rng.normal(0, 0.2, len(t)),
            "relative_humidity_2m": 72 - 12 * daily + rng.normal(0, 1, len(t)),
            "rain": rng.gamma(0.2, 0.3, len(t)),
            "wind_speed_10m": 12 + 2 * np.sin(2 * np.pi * t / 72) + rng.normal(0, 0.3, len(t)),
            "wind_direction_10m": (15 + 60 * np.sin(2 * np.pi * t / 96)) % 360,
            "surface_pressure": 1009 + 2 * np.sin(2 * np.pi * t / 120) + rng.normal(0, 0.3, len(t)),
        }
    )


@pytest.fixture
def config():
    # A smaller forest keeps these behavioral tests fast while retaining a fixed seed.
    return AnalysisConfig(n_estimators=80, random_state=42)


def test_fft_recovers_daily_temperature_cycle(weather, config):
    t = np.arange(len(weather))
    weather["temperature_2m"] = 27 + 5 * np.sin(2 * np.pi * t / 24)
    result = analyze(weather, config)
    spectrum = result.spectrum.loc[result.spectrum["variable"].eq("temperature_2m")]
    assert not spectrum.empty
    strongest = spectrum.sort_values("power", ascending=False).iloc[0]
    assert float(strongest["period_hours"]) == pytest.approx(24, abs=0.75)


def test_changing_future_data_cannot_change_past_results(weather, config):
    first = analyze(weather, config)
    altered = weather.copy(deep=True)
    changed_from = len(weather) - 48
    altered.loc[changed_from:, "temperature_2m"] += 12
    altered.loc[changed_from:, "rain"] += 20
    altered.loc[changed_from:, "surface_pressure"] -= 30
    second = analyze(altered, config)

    # Includes baselines, fitted scales, scores, decisions, and quality indicators.
    # A global FFT, centered smoother, backward fill, or global scaler breaks this.
    cutoff = weather.loc[changed_from, "date"]
    before_first = first.hourly.loc[first.hourly["date"] < cutoff].reset_index(drop=True)
    before_second = second.hourly.loc[second.hourly["date"] < cutoff].reset_index(drop=True)
    assert_frame_equal(before_first, before_second, check_exact=False, atol=1e-12, rtol=1e-12)
    assert_frame_equal(first.spectrum, second.spectrum, check_exact=False, atol=1e-12, rtol=1e-12)


def test_strong_heldout_multivariate_anomalies_are_detected(weather, config):
    anomalous_rows = np.arange(780, 786)
    weather.loc[anomalous_rows, "temperature_2m"] = 48
    weather.loc[anomalous_rows, "relative_humidity_2m"] = 99
    weather.loc[anomalous_rows, "rain"] = 60
    weather.loc[anomalous_rows, "wind_speed_10m"] = 100
    weather.loc[anomalous_rows, "surface_pressure"] = 950
    result = analyze(weather, config)
    event_dates = weather.loc[anomalous_rows, "date"]
    events = result.hourly.loc[result.hourly["date"].isin(event_dates)]
    assert events["score_valid"].all()
    assert events["is_anomaly"].mean() >= 0.8
    assert (events["anomaly_score"] > 0).mean() >= 0.8
    quiet = result.hourly.loc[
        result.hourly["split"].eq("evaluation") & ~result.hourly["date"].isin(event_dates)
    ]
    assert events["anomaly_score"].median() > quiet["anomaly_score"].median()
    assert not result.hourly.loc[result.hourly["split"].eq("train"), "is_anomaly"].any()


def test_wind_wraparound_is_a_small_physical_change(weather, config):
    weather.loc[[780, 781], "wind_speed_10m"] = 10
    weather.loc[780, "wind_direction_10m"] = 359
    weather.loc[781, "wind_direction_10m"] = 1
    result = analyze(weather, config)
    pair = result.hourly.set_index("date").loc[weather.loc[[780, 781], "date"]]
    # Circular direction changes by 2 degrees, so the vector displacement is small.
    du = pair["wind_u"].iloc[1] - pair["wind_u"].iloc[0]
    dv = pair["wind_v"].iloc[1] - pair["wind_v"].iloc[0]
    assert np.hypot(du, dv) < 0.4
    assert "wind_direction_10m_residual_z" not in result.hourly.columns


def test_missing_hour_is_reinserted_and_never_scored(weather, config):
    missing_time = weather.loc[780, "date"]
    result = analyze(weather.drop(index=780), config)
    assert len(result.hourly) == len(weather)
    row = result.hourly.set_index("date").loc[missing_time]
    assert not bool(row["quality_ok"])
    assert not bool(row["score_valid"])
    assert not bool(row["is_anomaly"])
    assert pd.isna(row["anomaly_score"])
    assert row["alert_level"] == "data_quality"

    # Rain accumulation uses a 24-hour context, so fabricated gap fills must
    # not leak into the following 23 decisions. Scoring resumes after recovery.
    indexed = result.hourly.set_index("date")
    affected_times = weather.loc[780:803, "date"]
    assert not indexed.loc[affected_times, "score_valid"].any()
    assert indexed.loc[affected_times, "anomaly_score"].isna().all()
    assert bool(indexed.loc[weather.loc[804, "date"], "score_valid"])


def test_invalid_numeric_value_is_flagged_and_not_scored(weather, config):
    weather["temperature_2m"] = weather["temperature_2m"].astype(object)
    weather.loc[780, "temperature_2m"] = "invalid sensor value"
    result = analyze(weather, config)
    row = result.hourly.set_index("date").loc[weather.loc[780, "date"]]
    assert not bool(row["quality_ok"])
    assert not bool(row["score_valid"])
    assert pd.isna(row["anomaly_score"])


def test_valid_hourly_timestamps_support_second_precision(weather, config):
    expected = analyze(weather, config)
    weather["date"] = weather["date"].dt.as_unit("s")
    actual = analyze(weather, config)
    np.testing.assert_allclose(
        actual.hourly["anomaly_score"], expected.hourly["anomaly_score"], equal_nan=True
    )
    assert actual.hourly["is_anomaly"].tolist() == expected.hourly["is_anomaly"].tolist()


def test_too_short_history_is_rejected_clearly(weather, config):
    with pytest.raises(ValueError) as error:
        analyze(weather.iloc[:48].copy(), config)
    assert len(str(error.value)) >= 15


def test_missing_required_variable_is_rejected_clearly(weather, config):
    with pytest.raises(ValueError) as error:
        analyze(weather.drop(columns="rain"), config)
    assert "rain" in str(error.value)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("train_fraction", 0.1),
        ("train_fraction", 0.95),
        ("train_fraction", np.nan),
        ("train_fraction", "0.7"),
        ("contamination", 0),
        ("contamination", 0.6),
        ("contamination", np.nan),
        ("contamination", "0.03"),
        ("top_k", 0),
        ("top_k", 11),
        ("top_k", 1.5),
        ("top_k", True),
        ("min_period_hours", 1),
        ("min_period_hours", "6"),
        ("max_period_hours", np.inf),
        ("max_period_hours", 5),
        ("n_estimators", 9),
        ("n_estimators", 2001),
        ("n_estimators", 80.5),
    ],
)
def test_invalid_config_has_clear_validation_error(name, value):
    with pytest.raises(ValueError) as error:
        AnalysisConfig(**{name: value}).validate()
    assert len(str(error.value)) >= 15
