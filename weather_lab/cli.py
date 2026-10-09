"""Command-line interface for reproducible FFT + Isolation Forest research."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from weather_lab.analysis import AnalysisConfig, analyze
from weather_lab.data import fetch_weather, make_demo_data
from weather_lab.reporting import export_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phân tích chu kỳ thời tiết bằng FFT và phát hiện bất thường bằng Isolation Forest.",
        epilog="Mặc định lấy dữ liệu Open-Meteo. Dùng --demo để chạy không cần mạng với dữ liệu mô phỏng.",
    )
    parser.add_argument("--demo", action="store_true", help="Sử dụng dữ liệu mô phỏng, không gọi API.")
    parser.add_argument("--demo-days", type=int, default=40, help="Số ngày mô phỏng (mặc định: 40).")
    parser.add_argument("--start", default="2026-09-01", help="Ngày bắt đầu, YYYY-MM-DD.")
    parser.add_argument("--end", default="2026-10-01", help="Ngày kết thúc bao gồm, YYYY-MM-DD.")
    parser.add_argument("--latitude", type=float, default=16.0678, help="Vĩ độ (mặc định: Đà Nẵng, 16.0678).")
    parser.add_argument("--longitude", type=float, default=108.2208, help="Kinh độ (mặc định: Đà Nẵng, 108.2208).")
    parser.add_argument("--timezone", default="Asia/Ho_Chi_Minh", help="Múi giờ IANA của đầu ra.")
    parser.add_argument("--train-fraction", type=float, default=0.7, help="Tỷ lệ đầu chuỗi dành cho huấn luyện (mặc định: 0.7).")
    parser.add_argument("--contamination", type=float, default=0.03, help="Tỷ lệ ngoại lệ giả định khi huấn luyện (mặc định: 0.03).")
    parser.add_argument("--top-k", type=int, default=3, help="Số chu kỳ Fourier tối đa mỗi biến (mặc định: 3).")
    parser.add_argument("--min-period-hours", type=float, default=6.0, help="Chu kỳ FFT tối thiểu (giờ).")
    parser.add_argument("--max-period-hours", type=float, default=168.0, help="Chu kỳ FFT tối đa (giờ).")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống cho mô phỏng và mô hình.")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Thư mục lưu CSV, JSON và báo cáo HTML.")
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache"), help="Thư mục cache Open-Meteo.")
    return parser


def main(argv: list[str] | None = None) -> int:
    # Windows redirected terminals otherwise use legacy code pages for Vietnamese output.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    try:
        config = AnalysisConfig(
            train_fraction=args.train_fraction, contamination=args.contamination, top_k=args.top_k,
            min_period_hours=args.min_period_hours, max_period_hours=args.max_period_hours,
            random_state=args.seed,
        )
        config.validate()
        if args.demo:
            print("Nguồn: DỮ LIỆU MÔ PHỎNG — không phải thời tiết thực tại Đà Nẵng.")
            frame = make_demo_data(days=args.demo_days, seed=args.seed, timezone=args.timezone)
        else:
            print(f"Nguồn: Open-Meteo · {args.latitude:.4f}, {args.longitude:.4f} · {args.start} → {args.end}")
            frame = fetch_weather(
                latitude=args.latitude, longitude=args.longitude, start_date=args.start,
                end_date=args.end, timezone=args.timezone, cache_dir=str(args.cache_dir),
            )
        result = analyze(frame, config)
        paths = export_result(result, args.output)
        hourly = result.hourly
        evaluation = hourly["split"].eq("evaluation")
        valid = evaluation & hourly["score_valid"].fillna(False)
        print(f"Đã phân tích {len(hourly):,} giờ; {int(valid.sum()):,} giờ đánh giá hợp lệ.")
        print(f"Giờ bất thường: {int(hourly['is_anomaly'].fillna(False).sum()):,} · Đợt cảnh báo: {len(result.alerts):,}")
        print("Tệp kết quả:")
        for name, path in paths.items():
            print(f"  {name}: {Path(path).resolve()}")
        print("Điểm bất thường không phải xác suất thiên tai. Công cụ phục vụ nghiên cứu.")
        return 0
    except KeyboardInterrupt:
        print("Đã dừng phân tích.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
