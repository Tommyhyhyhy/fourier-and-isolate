"""Causal, chronological anomaly research. No fitting uses evaluation values."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
from numbers import Integral, Real
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks
from sklearn.ensemble import IsolationForest

VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "rain", "wind_speed_10m",
    "wind_direction_10m", "surface_pressure",
)
SIGNALS = tuple(v for v in VARIABLES if v != "wind_direction_10m") + ("wind_u", "wind_v")
LABELS = {
    "temperature_2m": "Nhiệt độ", "relative_humidity_2m": "Độ ẩm",
    "rain": "Mưa", "wind_speed_10m": "Tốc độ gió", "surface_pressure": "Áp suất",
    "wind_u": "Thành phần gió Đông–Tây", "wind_v": "Thành phần gió Bắc–Nam",
}
UNITS = {"temperature_2m": "°C", "relative_humidity_2m": "%", "rain": "mm",
         "wind_speed_10m": "km/h", "surface_pressure": "hPa", "wind_u": "km/h", "wind_v": "km/h"}
# Prevent near-constant series from yielding arbitrarily large standardized residuals.
SCALE_FLOORS = {"temperature_2m": .2, "relative_humidity_2m": 1., "rain": .2,
                "wind_speed_10m": .5, "surface_pressure": .3, "wind_u": .5, "wind_v": .5}
ALERT_COLUMNS = ["start", "end", "duration_hours", "level", "max_anomaly_score", "reasons"]


@dataclass(frozen=True)
class AnalysisConfig:
    train_fraction: float = .7
    contamination: float = .03
    top_k: int = 3
    min_period_hours: float = 6.
    max_period_hours: float = 168.
    n_estimators: int = 300
    random_state: int = 42

    def validate(self) -> None:
        for field in ("train_fraction", "contamination", "min_period_hours", "max_period_hours"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"Tham số {field} phải là một số hữu hạn.")
        if not .2 <= self.train_fraction <= .9:
            raise ValueError("Tỷ lệ huấn luyện phải từ 0.2 đến 0.9.")
        if not 0 < self.contamination <= .5:
            raise ValueError("contamination phải lớn hơn 0 và không quá 0.5.")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, Integral) or not 1 <= self.top_k <= 10:
            raise ValueError("top_k phải là số nguyên từ 1 đến 10.")
        if not (np.isfinite(self.min_period_hours) and np.isfinite(self.max_period_hours)
                and 2 <= self.min_period_hours < self.max_period_hours):
            raise ValueError("Miền chu kỳ phải thỏa 2 ≤ min_period_hours < max_period_hours.")
        if isinstance(self.n_estimators, bool) or not isinstance(self.n_estimators, Integral) or not 10 <= self.n_estimators <= 2000:
            raise ValueError("Số cây phải là số nguyên từ 10 đến 2000.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, Integral) or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state phải là số nguyên từ 0 đến 2^32 - 1.")


@dataclass
class AnalysisResult:
    hourly: pd.DataFrame
    spectrum: pd.DataFrame
    alerts: pd.DataFrame
    summary: dict[str, Any]
    metadata: dict[str, Any]


def _prepare(frame: pd.DataFrame, config: AnalysisConfig):
    missing = set(("date",) + VARIABLES) - set(frame.columns)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(sorted(missing))}.")
    if len(frame) < 2:
        raise ValueError("Cần ít nhất 168 giờ huấn luyện và 24 giờ đánh giá.")
    # Require an explicit timezone so naive local CSV times are never silently called UTC.
    try:
        dates = pd.DatetimeIndex(frame["date"])
    except (ValueError, TypeError) as exc:
        raise ValueError("Cột date phải là thời gian hợp lệ, cùng múi giờ.") from exc
    if dates.hasnans or dates.tz is None:
        raise ValueError("Cột date phải có múi giờ (ví dụ +07:00), không được để trống.")
    if dates.has_duplicates:
        raise ValueError("Dữ liệu có giờ trùng lặp; cần xử lý trước khi phân tích.")
    dates = dates.as_unit("ns")
    order = np.argsort(dates.asi8)
    dates = dates[order]
    if np.any((dates.asi8 - dates.asi8[0]) % pd.Timedelta(hours=1).value):
        raise ValueError("Các mốc thời gian phải nằm trên lưới cách đều 1 giờ.")
    n = int((dates[-1] - dates[0]) / pd.Timedelta(hours=1)) + 1
    if n > 3 * 366 * 24:
        raise ValueError("Một lần phân tích hỗ trợ tối đa 3 năm dữ liệu theo giờ.")
    index = pd.date_range(dates[0], dates[-1], freq="h")
    values = frame.iloc[order][list(VARIABLES)].copy()
    values.index = dates
    values = values.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = values.reindex(index)
    # Only hard physical/data-schema constraints; unusual but possible weather is retained.
    for col, lo, hi in [("relative_humidity_2m", 0, 100), ("rain", 0, np.inf),
                        ("wind_speed_10m", 0, np.inf), ("wind_direction_10m", 0, 360),
                        ("surface_pressure", 0, np.inf), ("temperature_2m", -273.15, np.inf)]:
        values.loc[~values[col].between(lo, hi), col] = np.nan
    quality_ok = values.notna().all(axis=1)
    n_train = int(n * config.train_fraction)
    if n_train < 168 or n - n_train < 24:
        raise ValueError("Cần ít nhất 168 giờ huấn luyện và 24 giờ đánh giá; hãy tăng khoảng ngày.")
    if (~quality_ok.iloc[:n_train]).mean() > .1:
        raise ValueError("Hơn 10% giờ huấn luyện thiếu hoặc sai dữ liệu; cần nguồn dữ liệu tốt hơn.")
    medians = values.iloc[:n_train].median()
    if medians.isna().any():
        raise ValueError("Có biến không có giá trị hợp lệ trong đoạn huấn luyện.")
    # Causal fill: no interpolation/backfill across the train/evaluation boundary.
    filled = values.ffill(limit=2).fillna(medians)
    # All derived features depend on at most 24 current/past hourly observations.
    context_ok = quality_ok.rolling(24, min_periods=24).sum().eq(24)
    if context_ok.iloc[:n_train].sum() < 144:
        raise ValueError("Cần ít nhất 144 giờ huấn luyện có đủ 24 giờ dữ liệu liên tục trước đó.")
    return values, filled, quality_ok, context_ok, n_train


def _harmonics(t: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    columns = [np.ones(len(t))]
    for frequency in frequencies:
        columns.extend([np.cos(2 * np.pi * frequency * t), np.sin(2 * np.pi * frequency * t)])
    return np.column_stack(columns)


def _fourier_baseline(series: np.ndarray, n_train: int, valid: np.ndarray,
                      variable: str, config: AnalysisConfig):
    train = series[:n_train]
    nfft = 8 * n_train
    frequencies = np.fft.rfftfreq(nfft, d=1.)
    window = np.hanning(n_train)
    transformed = np.fft.rfft(detrend(train, type="linear") * window, n=nfft)
    # One-sided power spectrum; zero padding interpolates peaks, not true resolution.
    power = 2 * np.abs(transformed) ** 2 / window.sum() ** 2
    if nfft % 2 == 0:
        power[-1] /= 2
    power[0] /= 2
    max_period = min(config.max_period_hours, n_train / 2)
    allowed = (frequencies >= 1 / max_period) & (frequencies <= 1 / config.min_period_hours)
    peaks, _ = find_peaks(power, distance=8)
    peaks = peaks[allowed[peaks]]
    # Ignore numerical noise for flat signals and sidelobes below 5% of the strongest peak.
    if len(peaks) and power[peaks].max() > 1e-12:
        peaks = peaks[power[peaks] >= .05 * power[peaks].max()]
        chosen = peaks[np.argsort(power[peaks])[::-1][:config.top_k]]
    else:
        chosen = np.array([], dtype=int)
    selected_freq = frequencies[chosen]
    design = _harmonics(np.arange(len(series)), selected_freq)
    x_train, y_train = design[:n_train][valid], train[valid]
    weights = np.ones(len(y_train))
    coeff = np.zeros(design.shape[1])
    # Robust harmonic regression reduces the effect of isolated training outliers.
    for _ in range(5):
        root_w = np.sqrt(weights)
        coeff = np.linalg.lstsq(x_train * root_w[:, None], y_train * root_w, rcond=None)[0]
        resid = y_train - x_train @ coeff
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), SCALE_FLOORS[variable])
        weights = np.minimum(1., 1.5 * scale / np.maximum(np.abs(resid), 1e-12))
    baseline = design @ coeff
    if variable in ("rain", "wind_speed_10m"):
        baseline = np.maximum(0, baseline)
    elif variable == "relative_humidity_2m":
        baseline = np.clip(baseline, 0, 100)
    spectrum = pd.DataFrame({
        "variable": variable, "period_hours": 1 / frequencies[allowed],
        "frequency_per_hour": frequencies[allowed], "power": power[allowed],
        "selected": np.isin(np.flatnonzero(allowed), chosen),
    })
    return baseline, spectrum


def _group_alerts(hourly: pd.DataFrame) -> pd.DataFrame:
    active = hourly["is_anomaly"]
    groups = active.ne(active.shift(fill_value=False)).cumsum()
    levels = {"watch": 1, "warning": 2, "high": 3}
    events = []
    for _, group in hourly.loc[active].groupby(groups[active]):
        peak = group.loc[group["anomaly_score"].idxmax()]
        events.append({"start": group["date"].iloc[0].isoformat(),
                       "end": group["date"].iloc[-1].isoformat(),
                       "duration_hours": len(group),
                       "level": max(group["alert_level"], key=levels.get),
                       "max_anomaly_score": float(group["anomaly_score"].max()),
                       "reasons": str(peak["reasons"])})
    return pd.DataFrame(events, columns=ALERT_COLUMNS)


def analyze(frame: pd.DataFrame, config: AnalysisConfig | None = None) -> AnalysisResult:
    """Fit on the chronological prefix; score later hours with a frozen model.

    A retrospective holdout experiment, not a forecast or a calibrated hazard classifier.
    Missing current/context observations are excluded from model fitting and scoring.
    """
    config = config or AnalysisConfig()
    config.validate()
    raw, filled, quality_ok, context_ok, n_train = _prepare(frame, config)
    signals = filled.drop(columns="wind_direction_10m").copy()
    radians = np.deg2rad(filled["wind_direction_10m"])
    # Meteorological wind direction is where wind comes FROM.
    signals["wind_u"] = -filled["wind_speed_10m"] * np.sin(radians)
    signals["wind_v"] = -filled["wind_speed_10m"] * np.cos(radians)
    hourly = raw.reset_index(names="date")
    hourly["quality_ok"] = quality_ok.to_numpy()
    hourly["score_valid"] = context_ok.to_numpy()
    hourly["split"] = np.where(np.arange(len(raw)) < n_train, "train", "evaluation")
    features = pd.DataFrame(index=signals.index)
    spectra = []
    residual_z = pd.DataFrame(index=signals.index)
    fit_valid = context_ok.iloc[:n_train].to_numpy()
    for col in SIGNALS:
        series = signals[col].to_numpy()
        baseline, spectrum = _fourier_baseline(series, n_train, quality_ok.iloc[:n_train].to_numpy(), col, config)
        residual = series - baseline
        train_resid = residual[:n_train][fit_valid]
        center = float(np.median(train_resid))
        scale = max(1.4826 * np.median(np.abs(train_resid - center)), SCALE_FLOORS[col])
        z = (residual - center) / scale
        hourly[f"{col}_baseline"] = baseline
        # Do not present repaired inputs as observed residuals.
        hourly[f"{col}_residual"] = np.where(quality_ok, residual, np.nan)
        hourly[f"{col}_residual_z"] = np.where(quality_ok, z, np.nan)
        if col in ("wind_u", "wind_v"):
            hourly[col] = np.where(quality_ok, series, np.nan)
        features[f"{col}_level"] = series
        features[f"{col}_residual"] = residual
        features[f"{col}_delta_1h"] = signals[col].diff().fillna(0)
        residual_z[col] = z
        spectra.append(spectrum)
    features["rain_sum_6h"] = filled["rain"].rolling(6, min_periods=1).sum()
    features["rain_sum_24h"] = filled["rain"].rolling(24, min_periods=1).sum()
    features["pressure_delta_3h"] = filled["surface_pressure"].diff(3).fillna(0)
    model = IsolationForest(n_estimators=config.n_estimators, contamination=config.contamination,
                            random_state=config.random_state, n_jobs=-1)
    model.fit(features.iloc[:n_train].loc[context_ok.iloc[:n_train]])
    valid = context_ok.to_numpy()
    score = np.full(len(hourly), np.nan)
    score[valid] = -model.decision_function(features.loc[context_ok])
    ref = np.sort(score[:n_train][fit_valid])
    percentile = np.full(len(hourly), np.nan)
    percentile[valid] = np.searchsorted(ref, score[valid], side="right") / len(ref) * 100
    hourly["anomaly_score"] = score
    hourly["anomaly_percentile"] = percentile
    evaluation = np.arange(len(hourly)) >= n_train
    abnormal = valid & evaluation & (score > 0)
    hourly["is_anomaly"] = abnormal
    persistence = pd.Series(abnormal).rolling(3, min_periods=1).sum().to_numpy()
    max_z = residual_z.abs().max(axis=1).to_numpy()
    levels = np.full(len(hourly), "normal", dtype=object)
    levels[abnormal] = "watch"
    levels[abnormal & (persistence >= 2) & (max_z >= 3)] = "warning"
    levels[abnormal & (persistence >= 3) & (max_z >= 6)] = "high"
    levels[~evaluation] = "training"
    levels[~valid] = "data_quality"
    hourly["alert_level"] = levels
    reasons = []
    for i in range(len(hourly)):
        if not valid[i]:
            reasons.append("Thiếu/sai dữ liệu hoặc chưa đủ 24 giờ dữ liệu liên tục; không chấm điểm.")
        elif abnormal[i]:
            top = residual_z.iloc[i].abs().nlargest(3).index
            parts = [f"{LABELS[c]}: lệch {hourly.at[i, c + '_residual']:+.2f} {UNITS[c]} "
                     f"({residual_z.iloc[i][c]:+.1f} độ lệch chuẩn bền vững)" for c in top]
            reasons.append("; ".join(parts) + ". Mô tả độ lệch, không phải giải thích nhân quả của mô hình.")
        else:
            reasons.append("")
    hourly["reasons"] = reasons
    spectrum = pd.concat(spectra, ignore_index=True)
    alerts = _group_alerts(hourly)
    summary = {
        "total_hours": len(hourly), "training_hours": n_train,
        "training_fit_hours": int(fit_valid.sum()), "evaluation_hours": len(hourly) - n_train,
        "scored_evaluation_hours": int((valid & evaluation).sum()),
        "invalid_observation_hours": int((~quality_ok).sum()),
        "unscored_hours": int((~valid).sum()), "anomaly_hours": int(abnormal.sum()),
        "anomaly_rate_evaluation": float(abnormal.sum() / max((valid & evaluation).sum(), 1)),
        "alert_events": len(alerts),
        "evaluation_start": hourly["date"].iloc[n_train].isoformat(),
        "highest_alert": max(alerts["level"], key={"watch": 1, "warning": 2, "high": 3}.get) if len(alerts) else "normal",
    }
    metadata = {
        "source": frame.attrs.get("source", {"name": "Dữ liệu do người dùng cung cấp"}),
        "config": asdict(config), "timezone": str(raw.index.tz),
        "units": {**UNITS, "wind_direction_10m": "°"},
        "software_versions": {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "openmeteo-requests", "plotly")},
        "input_sha256": hashlib.sha256(pd.util.hash_pandas_object(raw, index=True).values.tobytes()).hexdigest(),
        "training_start": hourly["date"].iloc[0].isoformat(),
        "training_end": hourly["date"].iloc[n_train - 1].isoformat(),
        "evaluation_start": summary["evaluation_start"], "evaluation_end": hourly["date"].iloc[-1].isoformat(),
        "fft_frequency_resolution_per_hour": 1 / n_train,
        "fft_zero_padding_factor": 8, "fft_window": "Hann; linear detrend on training only",
        "feature_names": list(features.columns), "model_offset": float(model.offset_),
        "selected_periods": spectrum.loc[spectrum["selected"], ["variable", "period_hours"]].to_dict("records"),
        "score_definition": "anomaly_score = -IsolationForest.decision_function; > 0 means anomalous",
        "percentile_definition": "Empirical rank among training scores; not a probability of danger",
        "alert_policy": {"watch": "IF score > 0", "warning": "IF score > 0, >=2/3 recent hours anomalous, max |robust residual z| >=3",
                         "high": "IF score > 0, 3/3 recent hours anomalous, max |robust residual z| >=6"},
        "limitations": ["Phân tích hồi cứu dữ liệu theo giờ; không dự báo thời tiết tương lai.",
                        "Ngưỡng nghiên cứu, chưa hiệu chuẩn với nhãn sự kiện hoặc cảnh báo khí tượng chính thức.",
                        "Một tháng dữ liệu chỉ mô tả chu kỳ ngắn, không chứng minh được mùa vụ hàng năm.",
                        "contamination đặt ngưỡng trên tập huấn luyện, không phải tỷ lệ bất thường chắc chắn của tương lai.",
                        "FFT nền tuần hoàn cố định có thể suy giảm khi thời tiết chuyển chế độ; cần đánh giá cuốn chiếu.",
                        "Giờ thiếu/sai và 23 giờ kế tiếp bị loại khỏi chấm điểm vì cửa sổ đặc trưng 24 giờ."],
    }
    return AnalysisResult(hourly, spectrum, alerts, summary, metadata)
