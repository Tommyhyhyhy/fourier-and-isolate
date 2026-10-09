"""Network-free ingestion contract and synthetic-data regression tests."""

from datetime import datetime
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from weather_lab.data import ARCHIVE_URL, HOURLY_VARIABLES, fetch_weather, make_demo_data


def fake_response(*, count=48, interval=3600, broken_column=None, all_missing=False):
    start = pd.Timestamp("2026-09-01", tz="Asia/Ho_Chi_Minh")
    values = [np.full(count, 100 + index, dtype=float) for index in range(6)]
    if broken_column is not None:
        values[broken_column] = values[broken_column][:-1]
    if all_missing:
        values[0][:] = np.nan
    hourly = SimpleNamespace(
        Interval=lambda: interval,
        VariablesLength=lambda: len(values),
        Time=lambda: int(start.timestamp()),
        TimeEnd=lambda: int(start.timestamp()) + count * 3600,
        Variables=lambda index: SimpleNamespace(ValuesAsNumpy=lambda: values[index]),
    )
    return SimpleNamespace(
        Hourly=lambda: hourly,
        Latitude=lambda: 16.125,
        Longitude=lambda: 108.25,
        Elevation=lambda: 5.0,
        Timezone=lambda: b"Asia/Ho_Chi_Minh",
        UtcOffsetSeconds=lambda: 25200,
    )


@pytest.fixture
def api(tmp_path):
    session = MagicMock()
    client = MagicMock()
    client.weather_api.return_value = [fake_response()]
    with patch("weather_lab.data.requests_cache.CachedSession", return_value=session) as cached:
        with patch("weather_lab.data.retry", return_value=session) as retry_fn:
            with patch("weather_lab.data.openmeteo_requests.Client", return_value=client):
                with patch("weather_lab.data.datetime") as clock:
                    clock.now.return_value = datetime(2026, 10, 9)
                    yield SimpleNamespace(
                        client=client,
                        session=session,
                        cached=cached,
                        retry=retry_fn,
                        args={"start_date": "2026-09-01", "end_date": "2026-09-02", "cache_dir": tmp_path / "cache"},
                    )


def test_archive_and_variable_order_timezone(api):
    frame = fetch_weather(**api.args)
    assert list(frame.columns) == ["date", *HOURLY_VARIABLES]
    assert len(frame) == 48
    assert frame.date.iloc[0].isoformat() == "2026-09-01T00:00:00+07:00"
    assert frame.date.iloc[-1].isoformat() == "2026-09-02T23:00:00+07:00"
    assert frame.date.dt.tz is not None
    for index, variable in enumerate(HOURLY_VARIABLES):
        np.testing.assert_array_equal(frame[variable], np.full(48, 100 + index))
        assert frame[variable].dtype == np.dtype("float64")
    args, kwargs = api.client.weather_api.call_args
    assert args[0] == ARCHIVE_URL
    assert kwargs["params"]["hourly"] == list(HOURLY_VARIABLES)
    assert kwargs["params"]["timezone"] == "Asia/Ho_Chi_Minh"
    assert kwargs["params"]["wind_speed_unit"] == "kmh"
    assert kwargs["timeout"] == (10, 60)
    assert frame.attrs["source"]["synthetic"] is False
    assert frame.attrs["source"]["grid_latitude"] != 16.0678
    json.dumps(frame.attrs["source"], allow_nan=False)
    api.retry.assert_called_once_with(api.session, retries=5, backoff_factor=0.2)
    api.session.close.assert_called_once()


@pytest.mark.parametrize(
    "bad_args,match",
    [
        ({"latitude": float("nan")}, "Vĩ độ"),
        ({"latitude": float("inf")}, "Vĩ độ"),
        ({"latitude": 90.1}, "Vĩ độ"),
        ({"latitude": True}, "Vĩ độ"),
        ({"longitude": -181}, "Kinh độ"),
        ({"longitude": None}, "Kinh độ"),
        ({"timezone": "made/up"}, "Múi giờ"),
        ({"start_date": "2026-02-30"}, "Ngày bắt đầu"),
        ({"start_date": "20260901"}, "YYYY-MM-DD"),
        ({"start_date": "2026-09-03"}, "trước hoặc bằng"),
        ({"start_date": "1939-01-01"}, "1940"),
        ({"start_date": "2020-01-01", "end_date": "2023-01-01"}, "3 năm"),
        ({"end_date": "2026-10-05"}, "2026-10-04"),
    ],
)
def test_invalid_query_never_calls_network(api, bad_args, match):
    args = {**api.args, **bad_args}
    with pytest.raises(ValueError, match=match):
        fetch_weather(**args)
    api.client.weather_api.assert_not_called()
    api.cached.assert_not_called()


@pytest.mark.parametrize(
    "response,match",
    [
        (fake_response(count=0), "rỗng"),
        (fake_response(count=24), "không khớp"),
        (fake_response(interval=1800), "3600"),
        (fake_response(broken_column=2), "rain"),
        (fake_response(all_missing=True), "hoàn toàn thiếu"),
    ],
)
def test_malformed_payloads_raise_actionable_errors(api, response, match):
    api.client.weather_api.return_value = [response]
    with pytest.raises(RuntimeError, match=match):
        fetch_weather(**api.args)


@pytest.mark.parametrize("responses", [[], [fake_response(), fake_response()]])
def test_wrong_location_count_rejected(api, responses):
    api.client.weather_api.return_value = responses
    with pytest.raises(RuntimeError, match="đúng một vị trí"):
        fetch_weather(**api.args)


def test_network_error_does_not_return_demo(api):
    api.client.weather_api.side_effect = ConnectionError("network unavailable")
    with pytest.raises(RuntimeError, match="Kiểm tra mạng"):
        fetch_weather(**api.args)
    api.session.close.assert_called_once()


def test_cache_directory_error_is_actionable(api):
    api.args["cache_dir"].write_text("a file, not a directory")
    with pytest.raises(RuntimeError, match="quyền ghi"):
        fetch_weather(**api.args)
    api.client.weather_api.assert_not_called()


def test_partial_missing_is_preserved_and_counted(api):
    response = fake_response()
    # Keep the same array instance because ValuesAsNumpy returns a view.
    response.Hourly().Variables(0).ValuesAsNumpy()[5] = np.nan
    api.client.weather_api.return_value = [response]
    frame = fetch_weather(**api.args)
    assert np.isnan(frame.temperature_2m.iloc[5])
    assert frame.attrs["source"]["missing_values"]["temperature_2m"] == 1


def test_demo_reproducible_and_labelled():
    frame = make_demo_data()
    duplicate = make_demo_data()
    pd.testing.assert_frame_equal(frame, duplicate)
    assert list(frame.columns) == ["date", *HOURLY_VARIABLES]
    assert len(frame) == 40 * 24
    source = frame.attrs["source"]
    assert source["synthetic"] is True
    assert source["kind"] == "synthetic-demo"
    assert source["seed"] == 42
    assert len(source["injected_timestamps"]) == 12
    assert min(source["injected_row_indices"]) >= len(frame) * 0.8
    assert frame["rain"].ge(0).all()
    assert frame["wind_speed_10m"].ge(0).all()
    assert frame["relative_humidity_2m"].between(0, 100).all()
    assert frame["wind_direction_10m"].between(0, 360, inclusive="left").all()
    json.dumps(source, allow_nan=False)


def test_demo_seed_and_timezone():
    first = make_demo_data(days=10, seed=7, timezone="UTC")
    second = make_demo_data(days=10, seed=8, timezone="UTC")
    assert not np.array_equal(first.temperature_2m, second.temperature_2m)
    assert str(first.date.dt.tz) == "UTC"
    assert len(first) == 240


@pytest.mark.parametrize("days", [0, 6, 1096, True, 14.5])
def test_demo_invalid_days(days):
    with pytest.raises(ValueError, match="Số ngày demo"):
        make_demo_data(days=days)
