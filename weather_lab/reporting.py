"""Portable research exports. The HTML report embeds Plotly for offline viewing."""
from __future__ import annotations

from html import escape
import json
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio

from .analysis import AnalysisResult, LABELS, SIGNALS, UNITS


def _layout(fig: go.Figure, title: str, ytitle: str = "") -> None:
    fig.update_layout(title=title, template="plotly_white", height=440,
                      font=dict(family="Arial, sans-serif", color="#203d48"),
                      margin=dict(l=65, r=30, t=90, b=55),
                      legend=dict(orientation="h", y=1.08),
                      hovermode="x unified", yaxis_title=ytitle)


def build_html_report(result: AnalysisResult) -> str:
    hourly = result.hourly
    abnormal = hourly["is_anomaly"]
    chart = go.Figure()
    buttons = []
    for i, variable in enumerate(SIGNALS):
        label, unit = LABELS[variable], UNITS[variable]
        chart.add_trace(go.Scatter(x=hourly["date"], y=hourly[variable], mode="lines",
                                  name=f"{label} ({unit})", visible=i == 0,
                                  line=dict(color="#227e8b", width=1.5)))
        chart.add_trace(go.Scatter(x=hourly["date"], y=hourly[f"{variable}_baseline"],
                                  mode="lines", name="Nền Fourier", visible=i == 0,
                                  line=dict(color="#b87927", width=2, dash="dash")))
        chart.add_trace(go.Scatter(x=hourly.loc[abnormal, "date"], y=hourly.loc[abnormal, variable],
                                  mode="markers", name="Bất thường đa biến", visible=i == 0,
                                  marker=dict(color="#dc5666", size=7)))
        visible = [j // 3 == i for j in range(len(SIGNALS) * 3)]
        buttons.append(dict(label=label, method="update", args=[{"visible": visible}, {"yaxis.title.text": unit}]))
    _layout(chart, "Diễn biến và chu kỳ nền", UNITS[SIGNALS[0]])
    chart.update_layout(updatemenus=[dict(buttons=buttons, x=1., xanchor="right", y=1.2, yanchor="top")])
    boundary = result.summary["evaluation_start"]
    chart.add_shape(type="line", x0=boundary, x1=boundary, y0=0, y1=1, xref="x", yref="paper",
                    line=dict(color="#718690", dash="dot"))
    chart.add_annotation(x=boundary, y=1, xref="x", yref="paper", text="Bắt đầu đánh giá", showarrow=False)
    scores = go.Figure()
    scores.add_trace(go.Scatter(x=hourly["date"], y=hourly["anomaly_score"], name="Điểm bất thường",
                               mode="lines", line=dict(color="#227e8b")))
    scores.add_trace(go.Scatter(x=hourly.loc[abnormal, "date"], y=hourly.loc[abnormal, "anomaly_score"],
                               name="Giờ bất thường", mode="markers", marker=dict(color="#dc5666", size=6)))
    scores.add_hline(y=0, line_dash="dash", line_color="#b87927")
    _layout(scores, "Điểm Isolation Forest", "−decision_function")
    spectra = go.Figure()
    subset = result.spectrum[result.spectrum["variable"].eq("temperature_2m")].sort_values("period_hours")
    spectra.add_trace(go.Scatter(x=subset["period_hours"], y=subset["power"], name="Nhiệt độ: phổ FFT",
                                line=dict(color="#227e8b")))
    selected = subset[subset["selected"]]
    spectra.add_trace(go.Scatter(x=selected["period_hours"], y=selected["power"], name="Đỉnh được chọn",
                                mode="markers", marker=dict(color="#b87927", size=10)))
    _layout(spectra, "Phổ FFT nhiệt độ trên tập huấn luyện", "Công suất (°C²)")
    spectra.update_xaxes(title="Chu kỳ (giờ)", type="log")
    config = {"displaylogo": False, "responsive": True}
    charts = [pio.to_html(fig, full_html=False, include_plotlyjs=True if i == 0 else False,
                          config=config) for i, fig in enumerate((chart, scores, spectra))]
    source = result.metadata.get("source", {})
    synthetic = source.get("synthetic", False)
    source_label = "DỮ LIỆU MÔ PHỎNG — không phải dữ liệu thời tiết thực" if synthetic else source.get("provider", source.get("name", "Nguồn người dùng"))
    s = result.summary
    metric_items = [("Giờ dữ liệu", s["total_hours"]), ("Giờ đánh giá hợp lệ", s["scored_evaluation_hours"]),
                    ("Giờ bất thường", s["anomaly_hours"]), ("Đợt bất thường", s["alert_events"])]
    cards = "".join(f'<div class="metric"><span>{escape(label)}</span><b>{value:,}</b></div>' for label, value in metric_items)
    alert_table = result.alerts.to_html(index=False, escape=True, border=0) if len(result.alerts) else "<p>Không phát hiện đợt bất thường trong các giờ đánh giá hợp lệ.</p>"
    if s["scored_evaluation_hours"] == 0:
        alert_table = "<p>Không có giờ đánh giá hợp lệ; chưa thể kết luận về bất thường.</p>"
    periods = result.spectrum[result.spectrum["selected"]][["variable", "period_hours", "power"]].copy()
    periods["variable"] = periods["variable"].map(LABELS)
    period_table = periods.to_html(index=False, float_format=lambda x: f"{x:.3f}", escape=True, border=0)
    limitations = "".join(f"<li>{escape(item)}</li>" for item in result.metadata["limitations"])
    metadata_json = escape(json.dumps({"summary": s, "metadata": result.metadata}, ensure_ascii=False, indent=2, allow_nan=False))
    return f'''<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Weather Lab · Báo cáo FFT + Isolation Forest</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6f6;color:#203d48;font:15px/1.6 Arial,sans-serif}}
main{{max-width:1240px;margin:auto;padding:45px 24px}}.eyebrow{{font-size:12px;letter-spacing:2px;color:#227e8b;font-weight:bold}}
h1{{font-size:38px;line-height:1.2;margin:15px 0}}h2{{font-size:23px;margin-top:0}}.subtitle{{color:#617780}}
.source{{border-left:4px solid #b87927;padding:12px 18px;background:#fff7e8;margin:22px 0}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:15px}}.metric,section{{background:white;border:1px solid #dce5e6;border-radius:12px;padding:24px}}
.metric span{{display:block;color:#617780;font-size:13px}}.metric b{{font-size:32px}}section{{margin:20px 0;overflow:hidden}}
.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #e3eaeb}}th{{background:#edf4f4}}td{{min-width:90px}}
.note{{font-size:13px;color:#617780}}pre{{white-space:pre-wrap;word-break:break-word;font-size:12px}}a{{color:#227e8b}}
@media(max-width:700px){{.metrics{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:29px}}main{{padding:24px 12px}}section{{padding:12px}}}}
</style></head><body><main>
<div class="eyebrow">WEATHER LAB / BÁO CÁO NGHIÊN CỨU</div>
<h1>Chu kỳ thời tiết &amp; những giờ khác thường</h1>
<p class="subtitle">Fourier (FFT) + Isolation Forest · Múi giờ {escape(result.metadata['timezone'])}</p>
<div class="source">{escape(str(source_label))}<br>{escape(str(hourly['date'].iloc[0]))} → {escape(str(hourly['date'].iloc[-1]))}</div>
<div class="metrics">{cards}</div>
<section>{charts[0]}<p class="note">Chọn biến ở góc trên biểu đồ. Nền chỉ học từ đoạn huấn luyện. Chấm đỏ là bất thường đa biến; không xác định riêng biến đang xem là nguyên nhân.</p></section>
<section>{charts[1]}<p class="note">Điểm &gt; 0 vượt ngưỡng mô hình. Chỉ các giờ thuộc tập đánh giá được tạo cảnh báo. Percentile là thứ hạng so với tập huấn luyện, không phải xác suất nguy hiểm.</p></section>
<section><h2>Các đợt bất thường liên tiếp</h2><div class="table">{alert_table}</div>
<p class="note">watch: theo dõi; warning: bất thường rõ; high: bất thường cao. Quy tắc kết hợp điểm mô hình, sự lặp lại trong 3 giờ và độ lệch phần dư; đây là mức thống kê phục vụ nghiên cứu.</p></section>
<section>{charts[2]}<h2>Chu kỳ được chọn của từng biến</h2><div class="table">{period_table}</div>
<p class="note">Cửa sổ Hann và loại xu thế tuyến tính trước FFT. Độ phân giải tần số thực = 1/{s['training_hours']} chu kỳ/giờ; zero padding chỉ giúp nội suy vị trí đỉnh. Chỉ chọn chu kỳ có ít nhất hai vòng trong tập huấn luyện.</p></section>
<section><h2>Chất lượng &amp; giới hạn sử dụng</h2><p>{s['invalid_observation_hours']} giờ thiếu/sai dữ liệu; tổng cộng {s['unscored_hours']} giờ không được chấm điểm, gồm giai đoạn khởi tạo đặc trưng.</p><ul>{limitations}</ul></section>
<section><h2>Thông tin tái lập nghiên cứu</h2><details><summary>Xem tham số, nguồn dữ liệu và phương pháp</summary><pre>{metadata_json}</pre></details></section>
<p class="note">Tài liệu: <a href="https://open-meteo.com/en/docs/historical-weather-api">Open-Meteo</a> · <a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html">Isolation Forest</a> · <a href="https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html">NumPy FFT</a>.<br>Dữ liệu Open-Meteo: CC BY 4.0; cần giữ ghi công nguồn khi chia sẻ kết quả. Báo cáo này chạy được ngoại tuyến.</p>
</main></body></html>'''


def export_result(result: AnalysisResult, out_dir: str | Path) -> dict[str, Path]:
    directory = Path(out_dir).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / name for name in ("hourly.csv", "alerts.csv", "spectrum.csv", "summary.json", "report.html")}
    for name, frame in [("hourly.csv", result.hourly), ("alerts.csv", result.alerts), ("spectrum.csv", result.spectrum)]:
        frame.to_csv(paths[name], index=False, encoding="utf-8-sig")
    paths["summary.json"].write_text(json.dumps({"summary": result.summary, "metadata": result.metadata},
                                              ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    paths["report.html"].write_text(build_html_report(result), encoding="utf-8")
    return paths
