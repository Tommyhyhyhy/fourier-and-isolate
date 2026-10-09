"""Portable, inspectable export contracts without a browser or network access."""

from dataclasses import replace
from html.parser import HTMLParser
import json

import pandas as pd
import pytest

from weather_lab.analysis import AnalysisConfig, analyze
from weather_lab.data import make_demo_data
from weather_lab.reporting import build_html_report, export_result


class ReportMarkup(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.script_sources = []
        self.script_count = 0

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.script_count += 1
            source = dict(attrs).get("src")
            if source:
                self.script_sources.append(source)


@pytest.fixture(scope="module")
def report_result():
    return analyze(make_demo_data(), AnalysisConfig(n_estimators=80))


@pytest.fixture(scope="module")
def exported(report_result, tmp_path_factory):
    return export_result(report_result, tmp_path_factory.mktemp("research-export"))


def test_export_produces_all_five_nonempty_artifacts(exported):
    expected = {"hourly.csv", "alerts.csv", "spectrum.csv", "summary.json", "report.html"}
    assert set(exported) == expected
    for name, path in exported.items():
        assert path.name == name
        assert path.is_file()
        assert path.stat().st_size > 0


def test_hourly_csv_preserves_timezone_rows_and_missing_scores(exported, report_result):
    hourly = pd.read_csv(exported["hourly.csv"], encoding="utf-8-sig")
    assert list(hourly.columns) == list(report_result.hourly.columns)
    assert len(hourly) == len(report_result.hourly) == 40 * 24
    assert hourly["date"].str.endswith("+07:00").all()
    pd.testing.assert_series_equal(
        pd.to_datetime(hourly["date"], utc=True),
        report_result.hourly["date"].dt.tz_convert("UTC"),
        check_names=False,
    )
    assert hourly["is_anomaly"].tolist() == report_result.hourly["is_anomaly"].tolist()
    assert hourly["anomaly_score"].isna().tolist() == report_result.hourly["anomaly_score"].isna().tolist()


def test_alert_and_spectrum_csv_preserve_research_schema(exported, report_result):
    alerts = pd.read_csv(exported["alerts.csv"], encoding="utf-8-sig")
    spectrum = pd.read_csv(exported["spectrum.csv"], encoding="utf-8-sig")
    assert list(alerts.columns) == list(report_result.alerts.columns)
    assert len(alerts) == len(report_result.alerts)
    assert int(alerts["duration_hours"].sum()) == report_result.summary["anomaly_hours"]
    assert list(spectrum.columns) == list(report_result.spectrum.columns)
    assert len(spectrum) == len(report_result.spectrum)
    assert (spectrum["period_hours"] > 0).all()
    assert spectrum["selected"].tolist() == report_result.spectrum["selected"].tolist()


def test_summary_is_strict_json_with_reproducible_method_metadata(exported, report_result):
    def reject_nonfinite(value):
        raise AssertionError(f"JSON must not contain non-standard numeric token {value}")

    content = json.loads(exported["summary.json"].read_text(encoding="utf-8"), parse_constant=reject_nonfinite)
    assert set(content) == {"summary", "metadata"}
    assert content["summary"] == report_result.summary
    assert content["metadata"]["source"]["synthetic"] is True
    assert content["metadata"]["config"]["n_estimators"] == 80
    assert content["metadata"]["timezone"] == "Asia/Ho_Chi_Minh"
    assert pd.Timestamp(content["metadata"]["training_end"]) < pd.Timestamp(content["metadata"]["evaluation_start"])
    assert content["metadata"]["selected_periods"]
    assert content["metadata"]["feature_names"]
    assert content["metadata"]["limitations"]


def test_html_embeds_plotly_and_clearly_labels_simulated_data(exported):
    html = exported["report.html"].read_text(encoding="utf-8")
    markup = ReportMarkup()
    markup.feed(html)
    assert markup.script_count > 0
    assert markup.script_sources == []
    assert "plotly.js" in html
    assert "Plotly.newPlot" in html
    assert 'lang="vi"' in html
    assert "DỮ LIỆU MÔ PHỎNG" in html
    assert "không phải dữ liệu thời tiết thực" in html
    assert "không phải xác suất nguy hiểm" in html


def test_source_metadata_cannot_inject_html_script(report_result):
    source = {"synthetic": False, "provider": "<script>alert(1)</script>"}
    result = replace(report_result, metadata={**report_result.metadata, "source": source})
    html = build_html_report(result)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    expected_markup, actual_markup = ReportMarkup(), ReportMarkup()
    expected_markup.feed(build_html_report(report_result))
    actual_markup.feed(html)
    assert actual_markup.script_count == expected_markup.script_count
    assert actual_markup.script_sources == []


def test_empty_alert_export_keeps_csv_headers_and_readable_report(report_result, tmp_path):
    result = replace(
        report_result,
        alerts=report_result.alerts.iloc[:0].copy(),
        summary={**report_result.summary, "alert_events": 0, "highest_alert": "normal"},
    )
    exported = export_result(result, tmp_path / "no-alerts")
    alerts = pd.read_csv(exported["alerts.csv"], encoding="utf-8-sig")
    assert alerts.empty
    assert list(alerts.columns) == list(report_result.alerts.columns)
    html = exported["report.html"].read_text(encoding="utf-8")
    assert "Không phát hiện đợt bất thường trong các giờ đánh giá hợp lệ." in html
