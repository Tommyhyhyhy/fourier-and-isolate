# Weather Lab — FFT + Isolation Forest

Công cụ Python nghiên cứu chu kỳ thời tiết và phát hiện bất thường đa biến, có giao diện tiếng Việt, CLI và báo cáo HTML tương tác chạy ngoại tuyến. Mặc định dùng tọa độ Đà Nẵng **16.0678, 108.2208**, múi giờ **Asia/Ho_Chi_Minh**.

## Chạy ngay trên máy này

Môi trường `.venv` đã được cài trong `D:\nckh`. Mở PowerShell:

```powershell
cd /d D:\nckh
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Truy cập địa chỉ localhost mà Streamlit hiển thị. Chọn **Open-Meteo**, đặt ngày và bấm **Chạy phân tích** để dùng dữ liệu thật. Mặc định giao diện chọn **Dữ liệu mô phỏng** để thử luồng xử lý khi không có mạng; dữ liệu này luôn có nhãn riêng.

Phân tích bằng lệnh và xuất báo cáo:

```powershell
# Dữ liệu thật: ngày kết thúc được tính bao gồm cả ngày
.\.venv\Scripts\python.exe -m weather_lab --start 2026-09-01 --end 2026-10-01 --output outputs/danang-sep2026

# Dữ liệu mô phỏng với các sự kiện được chèn có chủ ý
.\.venv\Scripts\python.exe -m weather_lab --demo --output outputs/demo

# Xem các tùy chọn
.\.venv\Scripts\python.exe -m weather_lab --help
```

Kết quả chạy thật đã lưu ở `outputs/danang-sep2026/report.html`. Khoảng 01/09–01/10/2026 có **744 giờ**, chia **520 giờ đầu để huấn luyện**, **224 giờ cuối để đánh giá**. Với cấu hình mặc định lần chạy này, mô hình phát hiện **0 giờ bất thường trong phần đánh giá**. Các giờ huấn luyện không được gắn cảnh báo. Đây không phải kết luận về mức an toàn của thời tiết, cũng không có nghĩa toàn bộ tháng không có sự kiện đáng chú ý.

## Cài trên máy khác

Yêu cầu Python 3.10 trở lên. Bộ phiên bản đã kiểm thử trên Windows/Python 3.10 được ghi trong `requirements-lock.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Muốn tái lập đúng phiên bản đã chạy, cài `requirements-lock.txt` thay cho `requirements.txt`. Trên Linux/macOS dùng `.venv/bin/python` thay đường dẫn Windows. Không cần API key cho API công khai theo điều kiện sử dụng của Open-Meteo.

## Dữ liệu Open-Meteo

Dữ liệu lịch sử dùng `https://archive-api.open-meteo.com/v1/archive`, thay endpoint `/v1/forecast` trong ví dụ ban đầu. Mã vẫn dùng **openmeteo_requests**, **requests_cache** (cache 1 giờ) và **retry_requests** (5 lần thử lại).

| Biến | Ý nghĩa | Đơn vị |
|---|---|---|
| temperature_2m | Nhiệt độ ở 2 m | °C |
| relative_humidity_2m | Độ ẩm tương đối ở 2 m | % |
| rain | Lượng mưa tích lũy giờ theo API | mm |
| wind_speed_10m | Tốc độ gió ở 10 m | km/h |
| wind_direction_10m | Hướng gió đến từ | độ |
| surface_pressure | Áp suất bề mặt | hPa |

Đây là dữ liệu lịch sử dạng mô hình/tái phân tích **best match**, không phải đo trực tiếp tại trạm. Điểm lưới thực tế và cao độ có thể khác tọa độ yêu cầu; chúng được lưu trong metadata. API có thể kết hợp mô hình, nên khi nghiên cứu khí hậu dài hạn cần chọn bộ tái phân tích nhất quán.

Ngày bắt đầu/kết thúc tính theo múi giờ đã chọn; UNIX timestamp được đọc ở UTC rồi chuyển múi giờ đúng một lần. Công cụ kiểm tra đủ sáu biến, số điểm và bước 3.600 giây. Giới hạn thực hiện: mỗi lần tối đa 3 năm và ngày kết thúc cách hiện tại ít nhất 5 ngày để chừa độ trễ dữ liệu ERA5. Đây là lựa chọn bảo thủ của công cụ; một số mô hình archive có thể cập nhật sớm hơn. Khi API lỗi, công cụ báo lỗi và không thay bằng dữ liệu giả.

## Phương pháp

1. **Chia theo thời gian:** mặc định 70% đầu để huấn luyện, 30% sau để đánh giá; không xáo trộn. Cần ít nhất 168 giờ huấn luyện, 24 giờ đánh giá, và 144 giờ huấn luyện có đủ ngữ cảnh liên tục. Với cấu hình mặc định, nên lấy ít nhất 11 ngày, tốt hơn là vài tuần/tháng.
2. **Kiểm tra chất lượng:** đưa dữ liệu về lưới giờ, từ chối giờ trùng và bước thời gian lệch lưới. Giá trị thiếu, vô cực, phi số hoặc vi phạm miền vật lý được đánh dấu. Điền tiến tối đa 2 giờ rồi dùng trung vị *tập huấn luyện* để tạo chuỗi phục vụ FFT; không điền ngược từ tương lai. Hơn 10% giờ huấn luyện lỗi sẽ bị từ chối. Giờ lỗi và 23 giờ tiếp theo không được fit/chấm điểm vì các đặc trưng dùng cửa sổ 24 giờ. 23 giờ đầu cũng chưa có đủ ngữ cảnh.
3. **Hướng gió:** đổi sang `u = -speed × sin(direction)` và `v = -speed × cos(direction)`. Không lấy FFT trực tiếp trên góc 0°–360° để tránh gián đoạn giả.
4. **FFT trên tập huấn luyện:** loại xu thế tuyến tính, nhân cửa sổ Hann, dùng `numpy.fft.rfft`. Zero padding hệ số 8 giúp nội suy vị trí đỉnh, không làm tăng độ phân giải thực `1/N_train`. Chọn tối đa 3 đỉnh, chu kỳ mặc định 6–168 giờ, có ít nhất 2 vòng và công suất ít nhất 5% đỉnh mạnh nhất. Fit nền bằng hồi quy điều hòa có trọng số bền vững trên giờ huấn luyện hợp lệ; giữ hệ số cố định khi sang tập đánh giá. Nền gồm hằng số và sin/cos; không ngoại suy xu thế tuyến tính.
5. **Đặc trưng Isolation Forest:** giá trị hiện tại, phần dư `x(t) − nền(t)`, chênh lệch 1 giờ của 7 tín hiệu (5 biến vô hướng + u/v), tổng mưa 6/24 giờ và thay đổi áp suất 3 giờ. Mô hình gồm 300 cây, seed 42, contamination 0.03. Mô hình cây không cần chuẩn hóa đầu vào; median/MAD riêng trên tập huấn luyện được dùng để diễn giải độ lệch phần dư.
6. **Chấm điểm:** `anomaly_score = -decision_function`. Giá trị **> 0** là bất thường theo ngưỡng học được. `anomaly_percentile` là thứ hạng so với điểm tập huấn luyện, **không phải xác suất xảy ra thiên tai**. Contamination đặt ngưỡng theo tập huấn luyện; tỷ lệ cảnh báo ở tập đánh giá có thể là 0% hoặc khác 3%.
7. **Cảnh báo nghiên cứu:** chỉ tạo ở phần đánh giá hợp lệ, nhóm các giờ bất thường liền nhau thành đợt. Độ lệch mô tả ba tín hiệu lệch nhiều nhất; không phải giải thích nhân quả hay SHAP của Isolation Forest.

| Mức | Quy tắc tại giờ đang xét |
|---|---|
| watch / Theo dõi | Điểm Isolation Forest > 0 |
| warning / Bất thường rõ | Điểm > 0, ít nhất 2/3 giờ gần nhất bất thường và max \|robust residual z\| ≥ 3 |
| high / Bất thường cao | Điểm > 0, cả 3 giờ gần nhất bất thường và max \|robust residual z\| ≥ 6 |
| data_quality | Thiếu/sai dữ liệu hoặc chưa đủ ngữ cảnh; không phải cảnh báo thời tiết |

`robust residual z = (residual − median_train) / max(1.4826 × MAD_train, floor)`. Có sàn độ lệch theo đơn vị để chuỗi gần hằng số không tạo z vô hạn. Chi tiết nằm trong `weather_lab/analysis.py`. Mức cao nhất trong đợt được ghi vào bảng đợt; giờ kết thúc là giờ cuối cùng có cờ bất thường.

## Đầu ra và cấu trúc

Mỗi lần CLI chạy xuất 5 tệp; dùng thư mục khác nhau để giữ các thí nghiệm, vì chạy lại cùng thư mục sẽ ghi đè báo cáo cũ:

- `hourly.csv`: sáu biến, chất lượng, tập train/evaluation, nền, phần dư, điểm, percentile, mức cảnh báo và mô tả độ lệch. Giá trị đầu vào không hợp lệ được lưu là trống; không ghi số đã điền như thể là quan trắc.
- `alerts.csv`: các đợt liên tiếp; vẫn có tiêu đề khi không có cảnh báo.
- `spectrum.csv`: phổ huấn luyện, tần số, chu kỳ, công suất, cờ đỉnh được chọn.
- `summary.json`: kết quả, nguồn, tọa độ, đơn vị, tham số, mốc chia tập, phiên bản thư viện và SHA-256 của bảng đầu vào đã chuẩn hóa.
- `report.html`: báo cáo tương tác, nhúng Plotly nên đọc được không cần mạng.

CSV là UTF-8 BOM để dễ mở tiếng Việt bằng Excel. Timestamp giữ độ lệch múi giờ. Giao diện có nút tải cùng các đầu ra. Thay tham số sẽ đánh dấu kết quả cũ; dữ liệu/mô hình chỉ chạy lại khi bấm nút.

```text
app.py                    Giao diện Streamlit
weather_lab/data.py       Open-Meteo, kiểm tra phản hồi, demo tổng hợp
weather_lab/analysis.py    Chất lượng, FFT, đặc trưng, IF, cảnh báo
weather_lab/reporting.py   CSV / JSON / HTML
weather_lab/cli.py         Giao diện dòng lệnh
tests/                    Kiểm thử mô hình, API giả lập, UI và xuất báo cáo
```

## Kiểm thử và giới hạn

```powershell
.\.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
```

Các kiểm thử bao gồm nhận ra chu kỳ 24 giờ, dữ liệu tương lai không đổi kết quả quá khứ, phát hiện sự kiện đa biến chèn vào, hướng gió 359°→1°, giờ thiếu và quá trình phục hồi, đầu vào không hợp lệ, ánh xạ biến API và múi giờ, thao tác giao diện, xuất báo cáo. Dữ liệu mô phỏng dùng để kiểm tra hành vi; số giờ phát hiện trong demo không chứng minh độ chính xác trên thời tiết thực.

Đây là **phân tích hồi cứu có holdout**, chưa phải dự báo hoặc hệ thống cảnh báo thời gian thực. Nền Fourier là tham chiếu chu kỳ, không phải mô hình dự báo khí tượng đã kiểm định. Một tháng dữ liệu không xác lập chu kỳ khí hậu hàng năm. Mô hình có thể bỏ sót sự kiện hay gắn cờ chuyển chế độ tự nhiên; chưa có nhãn để tính precision/recall trên dữ liệu thật. Các quy tắc mức cảnh báo cần được hiệu chuẩn theo địa điểm và đối chiếu dữ liệu sự kiện trước khi dùng cho quyết định thực tế. Công cụ hiện hiển thị/xuất cảnh báo, không tự gửi email/SMS.

Để phát triển nghiên cứu tiếp: thu thập nhiều mùa, đánh giá cuốn chiếu với cửa sổ huấn luyện chỉ ở quá khứ, đối chiếu sự kiện đã xác minh và so sánh IF có/không có đặc trưng Fourier. Khi làm dự báo, cần tách rõ forecast với archive và thời điểm thông tin thực sự có sẵn.

## Nguồn tham khảo

- [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Điều khoản và ghi công Open-Meteo](https://open-meteo.com/en/terms)
- [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html)
- [NumPy rFFT](https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html)
- [SciPy periodogram: cửa sổ, detrend và phổ công suất](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)

Dữ liệu Open-Meteo được phân phối theo CC BY 4.0. Giữ ghi công Open-Meteo và nhà cung cấp theo điều khoản khi chia sẻ dữ liệu/kết quả; việc truy cập dịch vụ API có điều kiện sử dụng riêng. Dữ liệu demo do công cụ tự sinh.
