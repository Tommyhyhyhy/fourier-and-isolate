"""Exercise the actual Streamlit workflow, including explicit rerun semantics."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


def test_demo_workflow_and_stale_settings():
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    assert not app.exception
    assert app.radio[0].value == "Dữ liệu mô phỏng"
    assert any("Chạy phân tích" in item.value for item in app.info)
    assert "analysis_result" not in app.session_state

    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert len(app.metric) == 4
    assert any("DỮ LIỆU MÔ PHỎNG" in item.value for item in app.warning)
    result = app.session_state["analysis_result"]
    assert result.hourly["is_anomaly"].any()
    assert app.session_state["analysis_html"]
    initial_fingerprint = app.session_state["analysis_fingerprint"]

    # Choosing a chart variable is presentation-only and must not trigger ingestion or fitting.
    app.selectbox[0].select("rain").run()
    assert not app.exception
    assert app.session_state["analysis_fingerprint"] == initial_fingerprint
    assert app.session_state["analysis_result"].hourly.equals(result.hourly)

    # Changing scientific parameters exposes stale results until explicit submission.
    train_slider = next(item for item in app.slider if item.label == "Tỷ lệ dữ liệu huấn luyện")
    train_slider.set_value(0.75).run()
    assert not app.exception
    assert any("Thiết lập đã thay đổi" in item.value for item in app.warning)
    assert app.session_state["analysis_fingerprint"] == initial_fingerprint

    app.button[0].click().run()
    assert not app.exception
    assert not app.error
    assert app.session_state["analysis_fingerprint"] != initial_fingerprint
    assert not any("Thiết lập đã thay đổi" in item.value for item in app.warning)


def test_real_source_does_not_request_data_without_submit():
    app = AppTest.from_file(str(APP), default_timeout=45).run()
    app.radio[0].set_value("Open-Meteo").run()
    assert not app.exception
    assert "analysis_result" not in app.session_state
    assert not app.number_input[0].disabled
    assert not app.date_input[0].disabled
    assert any("Chạy phân tích" in item.value for item in app.info)
