"""Vietnamese local research dashboard for weather seasonality and anomalies."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from weather_lab.analysis import AnalysisConfig, analyze
from weather_lab.data import fetch_weather, make_demo_data
from weather_lab.reporting import build_html_report


VARIABLES = {
    "temperature_2m": ("Nhiệt độ", "°C"),
    "relative_humidity_2m": ("Độ ẩm tương đối", "%"),
    "rain": ("Lượng mưa", "mm"),
    "wind_speed_10m": ("Tốc độ gió", "km/h"),
    "surface_pressure": ("Áp suất bề mặt", "hPa"),
    "wind_u": ("Thành phần gió U", "km/h"),
    "wind_v": ("Thành phần gió V", "km/h"),
    "wind_direction_10m": ("Hướng gió", "°"),
}
LEVELS = {
    "normal": "Bình thường",
    "watch": "Theo dõi",
    "warning": "Bất thường rõ",
    "high": "Bất thường cao",
    "data_quality": "Cần kiểm tra dữ liệu",
    "training": "Dữ liệu huấn luyện",
}
CHART_COLORS = {"observed": "#247c8d", "baseline": "#b46e22", "anomaly": "#de5662"}


def _fingerprint(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8-sig")


def _style_chart(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=16, r=20, t=28, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, sans-serif", color="#29434d"),
        hovermode="x unified",
        legend=dict(orientation="h", y=1.15, x=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor="rgba(48,83,94,.12)", zeroline=False),
    )
    return fig


def _mark_evaluation(fig: go.Figure, hourly: pd.DataFrame) -> None:
    evaluation = hourly.loc[hourly["split"].eq("evaluation"), "date"]
    if not evaluation.empty:
        boundary = evaluation.iloc[0]
        fig.add_shape(
            type="line", x0=boundary, x1=boundary, y0=0, y1=1,
            xref="x", yref="paper", line=dict(color="#7b8d94", dash="dot"),
        )
        fig.add_annotation(
            x=boundary, y=1, xref="x", yref="paper", text="Bắt đầu đánh giá",
            showarrow=False, xanchor="left", yanchor="bottom",
            font=dict(size=11, color="#627980"),
        )


def _series_chart(hourly: pd.DataFrame, variable: str) -> go.Figure:
    label, unit = VARIABLES[variable]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly["date"], y=hourly[variable], name="Dữ liệu đầu vào", mode="lines",
        line=dict(color=CHART_COLORS["observed"], width=1.6), connectgaps=False,
    ))
    baseline = f"{variable}_baseline"
    if baseline in hourly:
        fig.add_trace(go.Scatter(
            x=hourly["date"], y=hourly[baseline], name="Chu kỳ nền Fourier", mode="lines",
            line=dict(color=CHART_COLORS["baseline"], width=2, dash="dash"), connectgaps=False,
        ))
    anomalies = hourly.loc[hourly["is_anomaly"].fillna(False)]
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["date"], y=anomalies[variable], name="Bất thường đa biến",
            mode="markers", marker=dict(color=CHART_COLORS["anomaly"], size=8),
            text=anomalies["reasons"], hovertemplate="%{x}<br>%{y:.2f}<br>%{text}<extra></extra>",
        ))
    _mark_evaluation(fig, hourly)
    fig.update_yaxes(title=f"{label} ({unit})")
    return _style_chart(fig, 420)


def _score_chart(hourly: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    valid = hourly["score_valid"].fillna(False)
    scores = hourly["anomaly_score"].where(valid)
    fig.add_trace(go.Scatter(
        x=hourly["date"], y=scores, mode="lines", name="Điểm bất thường",
        line=dict(color="#247c8d", width=1.8), connectgaps=False,
    ))
    flagged = hourly["is_anomaly"].fillna(False)
    fig.add_trace(go.Scatter(
        x=hourly.loc[flagged, "date"], y=hourly.loc[flagged, "anomaly_score"],
        mode="markers", name="Giờ bất thường", marker=dict(color="#de5662", size=7),
    ))
    fig.add_hline(y=0, line=dict(color="#b46e22", dash="dash"), annotation_text="Ngưỡng mô hình: 0")
    _mark_evaluation(fig, hourly)
    fig.update_yaxes(title="−decision_function · cao hơn = bất thường hơn")
    return _style_chart(fig, 320)


def main() -> None:
    st.set_page_config(page_title="Weather Lab · FFT + Isolation Forest", page_icon="🌤️", layout="wide")
    st.markdown("""
        <style>
        .stApp {background: #f4f7f7;}
        .block-container {padding-top: 4.5rem; max-width: 1500px;}
        [data-testid="stSidebar"] {background: #eaf0f0;}
        [data-testid="stMetric"] {background: #fff; border: 1px solid #dce5e6;
            padding: 18px; border-radius: 12px;}
        h1, h2, h3 {color: #183e4a;}
        .eyebrow {color: #247c8d; font-size: 12px; letter-spacing: 2px; font-weight: 700;}
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">WEATHER LAB / NGHIÊN CỨU THỜI TIẾT</div>', unsafe_allow_html=True)
    st.title("Hiểu chu kỳ. Nhận diện bất thường.")
    st.caption("FFT tách chu kỳ tự nhiên · Isolation Forest phát hiện độ lệch đa biến · Đánh giá theo thứ tự thời gian")

    with st.sidebar:
        st.header("Thiết lập nghiên cứu")
        source = st.radio("Nguồn dữ liệu", ["Dữ liệu mô phỏng", "Open-Meteo"], key="data_source")
        synthetic = source == "Dữ liệu mô phỏng"
        if synthetic:
            st.caption("Chạy thử không cần mạng. Các bất thường được chèn vào dữ liệu tổng hợp.")
        else:
            st.caption("Dữ liệu giờ lịch sử từ Open-Meteo. Nguồn cụ thể được lưu cùng báo cáo.")
        with st.form("run_analysis"):
            run = st.form_submit_button("Chạy phân tích", type="primary", use_container_width=True)
        latitude = st.number_input("Vĩ độ", min_value=-90.0, max_value=90.0, value=16.0678, format="%.4f", disabled=synthetic)
        longitude = st.number_input("Kinh độ", min_value=-180.0, max_value=180.0, value=108.2208, format="%.4f", disabled=synthetic)
        start = st.date_input("Ngày bắt đầu", value=date(2026, 9, 1), disabled=synthetic)
        end = st.date_input("Ngày kết thúc (bao gồm)", value=date(2026, 10, 1), disabled=synthetic)
        demo_days = st.slider("Số ngày mô phỏng", 14, 120, 40, disabled=not synthetic)
        st.caption("Múi giờ: Asia/Ho_Chi_Minh (UTC+7)")
        with st.expander("FFT và mô hình", expanded=False):
            train_fraction = st.slider("Tỷ lệ dữ liệu huấn luyện", 0.5, 0.8, 0.7, step=0.05)
            contamination = st.slider("Tỷ lệ ngoại lệ giả định khi huấn luyện", 0.005, 0.15, 0.03, step=0.005, format="%.3f")
            top_k = st.slider("Số chu kỳ Fourier tối đa / biến", 1, 8, 3)
            st.caption("Phần đầu chuỗi dùng để học; phần cuối dùng để đánh giá. FFT và mô hình chỉ học từ phần đầu.")
        settings = dict(
            source=source, latitude=latitude, longitude=longitude, start=str(start), end=str(end),
            demo_days=demo_days, train_fraction=train_fraction, contamination=contamination, top_k=top_k,
        )
        # Network and model work are gated by explicit submission; widget reruns only mark stale results.
        st.caption("Dữ liệu mô hình thời tiết và điểm bất thường phục vụ nghiên cứu. Đây không phải hệ thống cảnh báo thiên tai chính thức.")

    if run:
        if not synthetic and end < start:
            st.error("Ngày kết thúc phải bằng hoặc sau ngày bắt đầu.")
        else:
            try:
                with st.spinner("Đang chuẩn bị dữ liệu, phân tích FFT và huấn luyện Isolation Forest…"):
                    if synthetic:
                        frame = make_demo_data(days=demo_days, seed=42, timezone="Asia/Ho_Chi_Minh")
                    else:
                        frame = fetch_weather(
                            latitude=latitude, longitude=longitude, start_date=start.isoformat(),
                            end_date=end.isoformat(), timezone="Asia/Ho_Chi_Minh", cache_dir=".cache",
                        )
                    result = analyze(frame, AnalysisConfig(
                        train_fraction=train_fraction, contamination=contamination, top_k=top_k,
                    ))
                    html_report = build_html_report(result)
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_html"] = html_report
                    st.session_state["analysis_settings"] = settings
                    st.session_state["analysis_fingerprint"] = _fingerprint(settings)
            except Exception as exc:
                st.error(f"Không thể hoàn tất phân tích: {exc}")
                st.caption("Kiểm tra khoảng ngày, kết nối mạng và số lượng dữ liệu. Có thể dùng dữ liệu mô phỏng để thử công cụ.")

    result = st.session_state.get("analysis_result")
    if result is None:
        st.info("Chọn nguồn và nhấn **Chạy phân tích** để bắt đầu. Dữ liệu mô phỏng được chọn sẵn để thử toàn bộ quy trình.")
        first, second, third = st.columns(3)
        with first:
            st.subheader("01 / Dữ liệu theo giờ")
            st.write("Nhiệt độ, độ ẩm, mưa, tốc độ gió, hướng gió và áp suất. Kiểm tra thiếu dữ liệu trước khi chấm điểm.")
        with second:
            st.subheader("02 / Chu kỳ và độ lệch")
            st.write("FFT tìm chu kỳ ở tập huấn luyện. Isolation Forest học đặc trưng phần dư để nhận diện các kết hợp khác thường.")
        with third:
            st.subheader("03 / Cảnh báo có giải thích")
            st.write("Xem các giờ bất thường, yếu tố lệch nhiều nhất và nhóm cảnh báo liên tiếp. Xuất CSV, JSON và báo cáo HTML.")
        return

    active_settings = st.session_state["analysis_settings"]
    if st.session_state["analysis_fingerprint"] != _fingerprint(settings):
        st.warning("Thiết lập đã thay đổi. Kết quả bên dưới thuộc lần chạy trước; nhấn **Chạy phân tích** để cập nhật.")
    if active_settings["source"] == "Dữ liệu mô phỏng":
        st.warning("DỮ LIỆU MÔ PHỎNG — không phải thời tiết quan trắc hoặc dữ liệu thực tại Đà Nẵng.")
    else:
        st.info(f"Open-Meteo · tọa độ {active_settings['latitude']:.4f}, {active_settings['longitude']:.4f} · múi giờ Việt Nam")

    hourly = result.hourly
    evaluation = hourly["split"].eq("evaluation")
    valid_evaluation = evaluation & hourly["score_valid"].fillna(False)
    num_anomalies = int(hourly["is_anomaly"].fillna(False).sum())
    usable_count = int(valid_evaluation.sum())
    metrics = st.columns(4)
    metrics[0].metric("Giờ dữ liệu", f"{len(hourly):,}")
    metrics[1].metric("Giờ đánh giá hợp lệ", f"{usable_count:,}")
    metrics[2].metric("Giờ bất thường", f"{num_anomalies:,}", delta=f"{num_anomalies / usable_count:.1%} số giờ đánh giá" if usable_count else "Không đủ giờ hợp lệ", delta_color="off")
    metrics[3].metric("Đợt cảnh báo", f"{len(result.alerts):,}")
    st.caption(
        f"Khoảng dữ liệu: {hourly['date'].iloc[0]} → {hourly['date'].iloc[-1]} · "
        f"Huấn luyện {int((~evaluation).sum()):,} giờ · Đánh giá {int(evaluation.sum()):,} giờ"
    )

    overview, spectrum_tab, alerts_tab, export_tab = st.tabs(["Diễn biến thời tiết", "Phân tích FFT", "Cảnh báo & chất lượng", "Tải kết quả"])
    with overview:
        options = [name for name in VARIABLES if name in hourly]
        variable = st.selectbox("Biến thời tiết", options, format_func=lambda name: f"{VARIABLES[name][0]} ({VARIABLES[name][1]})")
        st.plotly_chart(_series_chart(hourly, variable), use_container_width=True)
        st.caption("Chấm đỏ là kết quả bất thường đa biến; không có nghĩa riêng biến đang xem gây ra cảnh báo. Đường nền được ước lượng từ tập huấn luyện.")
        st.subheader("Điểm bất thường")
        st.plotly_chart(_score_chart(hourly), use_container_width=True)
        st.caption("Điểm > 0 vượt ngưỡng Isolation Forest. Chỉ phần đánh giá tạo cảnh báo. Percentile là thứ hạng so với tập huấn luyện, không phải xác suất xảy ra thiên tai.")

    with spectrum_tab:
        st.subheader("Những chu kỳ học được từ dữ liệu")
        spectrum = result.spectrum
        if spectrum.empty:
            st.info("Không tìm được phổ chu kỳ hợp lệ trong tập huấn luyện.")
        else:
            spectrum_variables = list(spectrum["variable"].drop_duplicates())
            fft_variable = st.selectbox("Phổ của biến", spectrum_variables, format_func=lambda name: VARIABLES.get(name, (name, ""))[0])
            subset = spectrum.loc[spectrum["variable"].eq(fft_variable)].sort_values("period_hours")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=subset["period_hours"], y=subset["power"], mode="lines", name="Phổ công suất",
                line=dict(color="#247c8d", width=1.5),
            ))
            selected = subset.loc[subset["selected"].fillna(False)]
            fig.add_trace(go.Scatter(
                x=selected["period_hours"], y=selected["power"], mode="markers", name="Chu kỳ được chọn",
                marker=dict(color="#b46e22", size=10), hovertemplate="Chu kỳ %{x:.2f} giờ<br>Công suất %{y:.4g}<extra></extra>",
            ))
            fig.update_xaxes(title="Chu kỳ (giờ)", type="log")
            fig.update_yaxes(title="Công suất phổ")
            st.plotly_chart(_style_chart(fig), use_container_width=True)
            st.dataframe(selected[["variable", "period_hours", "frequency_per_hour", "power"]], use_container_width=True, hide_index=True)
        st.caption("Chuỗi ngắn chỉ cho thấy chu kỳ trong cửa sổ được phân tích; không đủ để kết luận về mùa khí hậu hoặc xu hướng dài hạn. Hướng gió được biểu diễn bằng hai thành phần vector để tránh điểm gián đoạn 0°/360°.")

    with alerts_tab:
        st.subheader("Các đợt bất thường liên tiếp")
        if not usable_count:
            st.warning("Không có giờ đánh giá hợp lệ để kết luận về bất thường. Kiểm tra chất lượng dữ liệu và chọn lại khoảng thời gian.")
        elif result.alerts.empty:
            st.success("Không có đợt bất thường được mô hình phát hiện trong phần đánh giá hợp lệ.")
        else:
            display_alerts = result.alerts.copy()
            if "level" in display_alerts:
                display_alerts["level"] = display_alerts["level"].map(LEVELS).fillna(display_alerts["level"])
            st.dataframe(display_alerts, use_container_width=True, hide_index=True)
        st.caption("Mức cảnh báo phản ánh độ khác thường thống kê và quy tắc của công cụ; không phải cấp độ rủi ro thiên tai được kiểm định.")
        st.subheader("Chất lượng dữ liệu")
        failed_quality = ~hourly["quality_ok"].fillna(False)
        st.write(f"{int(failed_quality.sum()):,} / {len(hourly):,} giờ có vấn đề chất lượng; {int((evaluation & ~hourly['score_valid'].fillna(False)).sum()):,} giờ đánh giá không được chấm điểm.")
        if failed_quality.any():
            quality_columns = [name for name in ["date", "split", "quality_ok", "score_valid", "alert_level", "reasons"] if name in hourly]
            st.dataframe(hourly.loc[failed_quality, quality_columns], use_container_width=True, hide_index=True)
        with st.expander("Dữ liệu và giải thích từng giờ"):
            st.dataframe(hourly, use_container_width=True, hide_index=True)
        with st.expander("Thông tin nguồn và phương pháp"):
            st.json(result.metadata)
            st.json(result.summary)

    with export_tab:
        st.subheader("Lưu một lần chạy nghiên cứu")
        st.write("Báo cáo HTML chứa biểu đồ và diễn giải. Các tệp CSV giữ lại dữ liệu giờ, phổ FFT và cảnh báo để phân tích tiếp.")
        left, right = st.columns(2)
        with left:
            st.download_button("Tải báo cáo HTML", st.session_state["analysis_html"], "weather_report.html", "text/html", use_container_width=True)
            st.download_button("Tải dữ liệu theo giờ · CSV", _csv_bytes(hourly), "hourly.csv", "text/csv", use_container_width=True)
            st.download_button("Tải cảnh báo · CSV", _csv_bytes(result.alerts), "alerts.csv", "text/csv", use_container_width=True)
        with right:
            st.download_button("Tải phổ Fourier · CSV", _csv_bytes(result.spectrum), "spectrum.csv", "text/csv", use_container_width=True)
            payload = json.dumps({"summary": result.summary, "metadata": result.metadata}, ensure_ascii=False, indent=2, default=str)
            st.download_button("Tải thông tin lần chạy · JSON", payload, "summary.json", "application/json", use_container_width=True)
        st.caption("Nguồn: Open-Meteo (https://open-meteo.com/). Dữ liệu mô phỏng luôn được ghi nhãn riêng.")


if __name__ == "__main__":
    main()
