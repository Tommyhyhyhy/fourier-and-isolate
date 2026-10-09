"""Open-Meteo historical ingestion and explicitly labelled synthetic examples."""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry


HOURLY_VARIABLES = (
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
)
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "rain": "mm",
    "wind_speed_10m": "km/h",
    "wind_direction_10m": "°",
    "surface_pressure": "hPa",
}


def _timezone(value: str) -> ZoneInfo:
    if not isinstance(value, str):
        raise ValueError("Múi giờ phải là tên IANA, ví dụ Asia/Ho_Chi_Minh.")
    try:
        return ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(
            f"Múi giờ không hợp lệ: {value!r}. Dùng tên IANA và cài gói tzdata."
        ) from exc


def _iso_date(value: str, name: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} phải có định dạng YYYY-MM-DD.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} không phải ngày hợp lệ: {value!r}.") from exc


def _coordinate(value: float, name: str, limit: float) -> float:
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} phải là số hữu hạn trong [-{limit:g}, {limit:g}].") from exc
    if isinstance(value, bool) or not math.isfinite(result) or abs(result) > limit:
        raise ValueError(f"{name} phải là số hữu hạn trong [-{limit:g}, {limit:g}].")
    return result


def _text(value: object) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def fetch_weather(
    latitude: float = 16.0678,
    longitude: float = 108.2208,
    start_date: str = "2026-09-01",
    end_date: str = "2026-10-01",
    timezone: str = "Asia/Ho_Chi_Minh",
    cache_dir: str | Path = ".cache",
) -> pd.DataFrame:
    """Fetch inclusive local calendar dates from the historical archive.

    Uses Open-Meteo best-match reanalysis, which is a modelled historical
    dataset, not direct station observations. This research tool conservatively
    requires dates at least five days old, matching ERA5 availability even
    though some archive models update sooner. Never substitutes synthetic data
    when the request fails. Missing values remain NaN for downstream handling.
    """
    lat = _coordinate(latitude, "Vĩ độ", 90)
    lon = _coordinate(longitude, "Kinh độ", 180)
    tz = _timezone(timezone)
    start = _iso_date(start_date, "Ngày bắt đầu")
    end = _iso_date(end_date, "Ngày kết thúc")
    if start > end:
        raise ValueError("Ngày bắt đầu phải trước hoặc bằng ngày kết thúc.")
    if start < date(1940, 1, 1):
        raise ValueError("Kho dữ liệu lịch sử Open-Meteo bắt đầu từ 1940-01-01.")
    if pd.Timestamp(end) >= pd.Timestamp(start) + pd.DateOffset(years=3):
        raise ValueError("Mỗi lần tải tối đa 3 năm; hãy chọn khoảng thời gian ngắn hơn.")
    latest = datetime.now(tz).date() - timedelta(days=5)
    if end > latest:
        raise ValueError(
            f"Dữ liệu lịch sử chỉ được tải đến {latest.isoformat()} trong công cụ này "
            "(chừa 5 ngày cho độ trễ ERA5). Hãy giảm ngày kết thúc."
        )

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": list(HOURLY_VARIABLES),
        "timezone": timezone,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    try:
        directory = Path(cache_dir).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        session = requests_cache.CachedSession(
            str(directory / "open_meteo"), expire_after=3600
        )
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "Không tạo được bộ nhớ đệm Open-Meteo. Hãy chọn thư mục cache có quyền ghi."
        ) from exc
    try:
        retry_session = retry(session, retries=5, backoff_factor=0.2)
        client = openmeteo_requests.Client(session=retry_session)
        responses = client.weather_api(ARCHIVE_URL, params=params, timeout=(10, 60))
    except Exception as exc:
        raise RuntimeError(
            "Không tải được dữ liệu lịch sử Open-Meteo. Kiểm tra mạng, khoảng ngày "
            f"và giới hạn API, sau đó thử lại. Chi tiết: {exc}"
        ) from exc
    finally:
        session.close()

    try:
        if len(responses) != 1:
            raise ValueError("API phải trả về đúng một vị trí.")
        response = responses[0]
        hourly = response.Hourly()
        if hourly is None or hourly.Interval() != 3600:
            raise ValueError("API không trả về chuỗi theo giờ (bước 3600 giây).")
        if hourly.VariablesLength() != len(HOURLY_VARIABLES):
            raise ValueError("API không trả về đủ sáu biến thời tiết đã yêu cầu.")
        if hourly.TimeEnd() <= hourly.Time():
            raise ValueError("API trả về chuỗi thời gian rỗng hoặc đảo ngược.")
        # FlatBuffers UNIX timestamps are UTC. Apply tz_convert exactly once;
        # adding UtcOffsetSeconds here would shift local times twice.
        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq="h",
            inclusive="left",
        ).tz_convert(tz)
        if len(times) == 0:
            raise ValueError("API trả về chuỗi thời gian rỗng.")
        expected = pd.date_range(
            pd.Timestamp(start, tz=tz),
            pd.Timestamp(end + timedelta(days=1), tz=tz),
            freq="h",
            inclusive="left",
        )
        if not times.equals(expected):
            raise ValueError("Khoảng thời gian API trả về không khớp các ngày đã yêu cầu.")
        values = {}
        for index, variable in enumerate(HOURLY_VARIABLES):
            array = np.asarray(hourly.Variables(index).ValuesAsNumpy(), dtype=float)
            if array.ndim != 1 or len(array) != len(times):
                raise ValueError(f"Số giá trị của {variable} không khớp chuỗi thời gian.")
            if np.isinf(array).any() or np.isnan(array).all():
                raise ValueError(f"Biến {variable} chứa vô cực hoặc hoàn toàn thiếu dữ liệu.")
            values[variable] = array
        frame = pd.DataFrame({"date": times, **values})
        frame.attrs["source"] = {
            "kind": "open-meteo-archive",
            "synthetic": False,
            "provider": "Open-Meteo",
            "dataset": "best-match historical reanalysis",
            "url": ARCHIVE_URL,
            "documentation_url": "https://open-meteo.com/en/docs/historical-weather-api",
            "requested_latitude": lat,
            "requested_longitude": lon,
            "grid_latitude": float(response.Latitude()),
            "grid_longitude": float(response.Longitude()),
            "elevation_m": float(response.Elevation()),
            "timezone": timezone,
            "api_timezone": _text(response.Timezone()),
            "utc_offset_seconds": int(response.UtcOffsetSeconds()),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "interval_seconds": 3600,
            "units": dict(UNITS),
            "missing_values": {name: int(frame[name].isna().sum()) for name in HOURLY_VARIABLES},
        }
        return frame
    except Exception as exc:
        raise RuntimeError(
            "Dữ liệu Open-Meteo không đáp ứng cấu trúc cần thiết. "
            f"Hãy thử khoảng ngày khác hoặc kiểm tra phản hồi API. Chi tiết: {exc}"
        ) from exc


def make_demo_data(
    days: int = 40, seed: int = 42, timezone: str = "Asia/Ho_Chi_Minh"
) -> pd.DataFrame:
    """Generate reproducible synthetic weather with late injected anomalies.

    The final 20% contains deliberately exaggerated rain/wind/pressure and
    temperature events for pipeline demonstrations, never real observations.
    """
    if isinstance(days, bool) or not isinstance(days, (int, np.integer)) or not 7 <= days <= 1095:
        raise ValueError("Số ngày demo phải là số nguyên từ 7 đến 1095.")
    tz = _timezone(timezone)
    rng = np.random.default_rng(seed)
    count = int(days) * 24
    times = pd.date_range("2026-09-01", periods=count, freq="h", tz=tz)
    hour = np.arange(count, dtype=float)
    daily = np.sin(2 * np.pi * (hour - 8) / 24)
    synoptic = np.sin(2 * np.pi * hour / (24 * 6.5))
    wet = rng.random(count) < (0.08 + 0.04 * (1 + synoptic))
    rain = np.where(wet, rng.gamma(1.3, 1.0, count), 0.0)
    frame = pd.DataFrame(
        {
            "date": times,
            "temperature_2m": 28 + 3.5 * daily + 0.8 * synoptic + rng.normal(0, 0.45, count),
            "relative_humidity_2m": np.clip(78 - 11 * daily + 4 * synoptic + rng.normal(0, 2.2, count), 30, 100),
            "rain": rain,
            "wind_speed_10m": np.maximum(0, 10 + 3 * daily + 1.8 * synoptic + rng.normal(0, 1.0, count)),
            "wind_direction_10m": (120 + 30 * np.sin(2 * np.pi * hour / 24) + 18 * synoptic + rng.normal(0, 7, count)) % 360,
            "surface_pressure": 1007 + 1.5 * synoptic + 0.7 * np.cos(2 * np.pi * hour / 12) + rng.normal(0, 0.22, count),
        }
    )
    storm_start = int(count * 0.83)
    storm_rows = np.arange(storm_start, min(storm_start + 8, count))
    frame.loc[storm_rows, "rain"] += 18 + rng.uniform(0, 12, len(storm_rows))
    frame.loc[storm_rows, "wind_speed_10m"] += 40 + rng.uniform(0, 12, len(storm_rows))
    frame.loc[storm_rows, "surface_pressure"] -= 16
    frame.loc[storm_rows, "relative_humidity_2m"] = 98
    frame.loc[storm_rows, "temperature_2m"] -= 5
    frame.loc[storm_rows, "wind_direction_10m"] = 285
    heat_start = int(count * 0.94)
    heat_rows = np.arange(heat_start, min(heat_start + 4, count))
    frame.loc[heat_rows, "temperature_2m"] += 12
    frame.loc[heat_rows, "relative_humidity_2m"] = 35
    injected = np.unique(np.concatenate([storm_rows, heat_rows]))
    frame.attrs["source"] = {
        "kind": "synthetic-demo",
        "synthetic": True,
        "provider": "Bộ sinh dữ liệu mô phỏng nội bộ — KHÔNG phải Open-Meteo",
        "dataset": "DỮ LIỆU GIẢ LẬP: chu kỳ ngày + nhiễu + bất thường được cài vào",
        "timezone": timezone,
        "start_date": times[0].date().isoformat(),
        "end_date": times[-1].date().isoformat(),
        "interval_seconds": 3600,
        "seed": int(seed),
        "units": dict(UNITS),
        "injected_timestamps": [times[int(index)].isoformat() for index in injected],
        "injected_row_indices": [int(index) for index in injected],
        "injection_start_fraction": 0.83,
    }
    return frame
