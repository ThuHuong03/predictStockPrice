# Dự báo xu hướng cổ phiếu VN30 bằng học máy và mô hình chuỗi thời gian

**Tài liệu độc lập** — dành cho giảng viên, đối tác, hoặc bất kỳ ai **chưa tham gia** phát triển repository này. Bạn không cần đọc mã nguồn trước; mọi khái niệm được giải thích theo trình tự. Phần cuối có **mục lệnh chạy** và **cấu trúc thư mục** để tái hiện kết quả.

---

## Mục lục

1. [Đọc nhanh 3 phút](#1-đọc-nhanh-3-phút-cho-người-bận)  
2. [Repository này là gì, không phải là gì](#2-repository-này-là-gì-không-phải-là-gì)  
3. [Bối cảnh bài toán và ký hiệu](#3-bối-cảnh-bài-toán-và-ký-hiệu)  
4. [Nguồn dữ liệu và universe cổ phiếu](#4-nguồn-dữ-liệu-và-universe-cổ-phiếu)  
5. [Luồng xử lý end-to-end (từ raw đến model)](#5-luồng-xử-lý-end-to-end-từ-raw-đến-model)  
6. [Nhãn mục tiêu: ba lớp TĂNG / SIDEWAY / GIẢM](#6-nhãn-mục-tiêu-ba-lớp-tăng--sideway--giảm)  
7. [Đặc trưng (feature engineering) — nhóm và ý nghĩa kinh tế](#7-đặc-trưng-feature-engineering--nhóm-và-ý-nghĩa-kinh-tế)  
8. [Meta-labeling kiểu AFML: EMA làm “primary”, triple-barrier làm nhãn phụ](#8-meta-labeling-kiểu-afml-ema-làm-primary-triple-barrier-làm-nhãn-phụ)  
9. [Chia train / validation / test theo thời gian](#9-chia-train--validation--test-theo-thời-gian)  
10. [Chọn bộ đặc trưng và kiểm tra chất lượng dữ liệu](#10-chọn-bộ-đặc-trưng-và-kiểm-tra-chất-lượng-dữ-liệu)  
11. [Các mô hình học máy — mô tả dài cho từng thuật toán](#11-các-mô-hình-học-máy--mô-tả-dài-cho-từng-thuật-toán)  
12. [Mô hình kinh tế lượng: AR(1) mean + GARCH(1,1) variance (joint)](#12-mô-hình-kinh-tế-lượng-ar1-mean--garch11-variance-joint)  
13. [Đánh giá: metric là gì và vì sao dùng](#13-đánh-giá-metric-là-gì-và-vì-sao-dùng)  
14. [Bảng xếp hạng, “best model”, và rolling-origin](#14-bảng-xếp-hạng-best-model-và-rolling-origin)  
15. [Suy diễn: fusion giữa ML và chuỗi thời gian](#15-suy-diễn-fusion-giữa-ml-và-chuỗi-thời-gian)  
16. [File đầu ra, Colab, web, và tái lập môi trường](#16-file-đầu-ra-colab-web-và-tái-lập-môi-trường)  
17. [Hạn chế khoa học và trách nhiệm sử dụng](#17-hạn-chế-khoa-học-và-trách-nhiệm-sử-dụng)  
18. [Câu hỏi thường gặp (FAQ)](#18-câu-hỏi-thường-gặp-faq)  
19. [Phụ lục: lệnh chạy và cấu trúc thư mục](#19-phụ-lục-lệnh-chạy-và-cấu-trúc-thư-mục)

---

## 1. Đọc nhanh 3 phút (cho người bận)

- **Mục tiêu kỹ thuật:** Dự đoán **nhãn xu hướng ngắn hạn** (tăng đủ mạnh / đi ngang / giảm đủ mạnh) cho các mã cổ phiếu được chọn trong **rổ VN30**, dựa trên lịch sử giá–khối lượng từ **Yahoo Finance**, và **so sánh** nhiều thuật toán học máy với một **baseline chuỗi thời gian có phương sai thay đổi** (AR(1)–GARCH joint).

- **Không phải:** Dự báo giá tuyệt đối chính xác từng đồng; không phải hệ thống giao dịch tự động hoàn chỉnh (thiếu chi phí giao dịch, thanh khoản thực, lệnh, rủi ro danh mục…).

- **Pipeline chính** (một file Python lớn): `stock_analysis_yfinance.py` — thu thập dữ liệu → tiền xử lý → chỉ báo kỹ thuật → tạo đặc trưng → huấn luyện nhiều mô hình → benchmark kinh tế lượng → lưu **bundle** (`analysis_bundle.joblib`).

- **Hai “chế độ” huấn luyện ML quan trọng:**  
  - **Đa lớp:** học trực tiếp ba nhãn `TANG` / `SIDEWAY` / `GIAM`.  
  - **Meta-labeling:** học **nhị phân** “tín hiệu theo quy tắc EMA có mang lại kết quả tốt theo triple-barrier hay không”, rồi khi suy diễn kết hợp lại với nhãn đa lớp — hướng tiếp cận gợi ý trong *Advances in Financial Machine Learning* (López de Prado).

- **Tài liệu báo cáo hình ảnh:** `bundle_chart_report.py` đọc bundle đã train và xuất biểu đồ vào `artifacts/colab_charts/`. Notebook `colab.ipynb` hướng dẫn chạy trên Google Colab sau khi upload bundle.

---

## 2. Repository này là gì, không phải là gì

### 2.1. Là gì

- Một **pipeline nghiên cứu / học tập** có thể chạy lại từ đầu đến cuối.
- Tập trung **đánh giá có kiểm soát**: chia mẫu theo thời gian, báo cáo độ chính xác / F1 / (tuỳ trường hợp) ROC-AUC trên tập **test** chưa dùng để chọn mô hình.
- Có **đường baseline kinh tế lượng** để tránh việc chỉ báo cáo “ML cao hơn không” mà không có mốc so sánh lý thuyết chuỗi thời gian.

### 2.2. Không phải

- **Không** phải chứng thư đăng ký giao dịch hay khuyến nghị đầu tư.
- **Không** đảm bảo lợi nhuận ngoài mẫu; hiệu năng quá khứ **không** hàm ý hiệu năng tương lai.
- **Không** mô phỏng đầy đủ ma sát thị trường (spread, impact, thuế, hạn chế vị thế…).

---

## 3. Bối cảnh bài toán và ký hiệu

### 3.1. Vì sao phân loại thay vì hồi quy giá?

Dự báo **log-return** hoặc **giá đóng cửa kỳ sau** là bài toán hồi quy cổ điển. Tuy nhiên, trong báo cáo nhóm và trình bày định hướng **quyết định** (“xu hướng lên/xuống/đi ngang”), phân loại ba lớp thường **dễ diễn giải** hơn và gắn với ngưỡng kinh tế (biến động nhỏ coi là không đổi đáng kể).

### 3.2. Ký hiệu thường gặp

| Ký hiệu | Ý nghĩa |
|--------|---------|
| \(t\) | Chỉ số thời gian (ngày giao dịch). |
| \(r_{t}\) hoặc `daily_return` | Lợi suất đơn giản hoặc log-return trong pipeline (code quy ước cụ thể). |
| **VN30** | Rổ tiêu chí lớn trên HOSE; universe đầu vào trước khi chọn Top mã. |
| **Top 10** | Một tập con động các mã được chọn theo tiêu chí hiệu suất lịch sử (mặc định ~5 năm trong pipeline chuẩn). |

---

## 4. Nguồn dữ liệu và universe cổ phiếu

### 4.1. Yahoo Finance (qua `yfinance`)

- Dữ liệu **OHLCV** (open, high, low, close, volume) theo ngày (hoặc tần suất pipeline hỗ trợ).
- **Ưu điểm:** miễn phí, dễ tái lập cho bài tập/học thuật.  
- **Hạn chế:** có thể điều chỉnh giá lịch sử khi có sự kiện doanh nghiệp; không thay thế dữ liệu sàn chính thức cho mục đích tuân thủ.

### 4.2. Universe và chọn mã

Pipeline nạp **danh sách VN30** cố định trong code (universe), sau đó **xếp hạng** hiệu suất dài hạn và giữ **Top N** (thường 10) để tập trung tài nguyên tính toán và báo cáo. Điều này có nghĩa kết quả benchmark là cho **tập mã đang được theo dõi**, không phải toàn thị trường.

---

## 5. Luồng xử lý end-to-end (từ raw đến model)

Hàm tổng quát là `run_full_pipeline` trong `stock_analysis_yfinance.py`. Trình tự **logic**:

1. **`collect_stock_data_yfinance`** — Tải dữ liệu cho universe, có **cache** parquet dưới `artifacts/cache/` để lần sau không phải gọi API lại (có thể bật làm mới).
2. **`preprocess_data`** — Xử lý giá trị thiếu, outlier cơ bản cho chuỗi đa mã.
3. **`run_eda`** — Thống kê mô tả, hình ảnh phân phối return, tương quan… → `artifacts/eda/`.
4. **`calculate_technical_indicators`** — EMA, RSI, MACD, dải Bollinger, OBV, VWAP, … trên `processed_data`.
5. **`add_hybrid_econometric_features`** — Fit **joint ARX(1)–GARCH(1,1)** theo cửa sổ mở rộng cho từng mã (thư viện `arch`), tạo các cột kiểu dự báo return / độ lệch / biến động. **Các cột hybrid này không được gộp vào `feature_cols` của ML** để tránh **rò rỉ** (leakage) từ chính quy trình econometric vào huấn luyện phân loại (đã tách rõ trong code: chỉ `core + regime + seasonal + market`).
6. **`prepare_features`** — Xây ma trận đặc trưng cuối, tạo `target_class`, và nếu bật `use_meta_labeling` thì tạo `primary_side`, triple-barrier, `meta_label`.
7. **`train_model`** — Gọi nhánh ML phù hợp + `train_econometric_models` + `build_benchmark_summary` + (tuỳ cấu hình) rolling đánh giá.
8. **`save_model`** — Ghi `analysis_bundle.joblib` chứa models, benchmark, cấu hình, v.v.

---

## 6. Nhãn mục tiêu: ba lớp TĂNG / SIDEWAY / GIẢM

### 6.1. Cách tạo nhãn (đa lớp)

Trên mỗi dòng (một mã, một ngày):

1. Lấy **giá đóng kỳ sau** (shift theo `target_shift`, mặc định 1 phiên).
2. Tính **phần trăm thay đổi** so với giá hiện tại → `next_return_pct`.
3. Áp hàm `_trend_label`:

   - Nếu \(|next\_return\_pct| <\) **`sideway_threshold`** (mặc định **0.75** phần trăm) → nhãn **SIDEWAY**.  
   - Nếu ≥ ngưỡng và dương → **TANG**.  
   - Nếu ≥ ngưỡng và âm → **GIAM**.

Ngưỡng **0.75%** một phiên là **quy ước kỹ thuật**: vùng “đi ngang” rộng thì lớp SIDEWAY nhiều hơn; hẹp thì ít hơn. Khi báo cáo khoa học, nên **luôn nêu rõ** ngưỡng này vì nó định nghĩa trực tiếp **độ khó** của bài toán và mức **mất cân bằng lớp**.

### 6.2. Nhãn phụ `target_binary`

Code còn lưu nhị phân “return kỳ sau > 0 hay không” — có thể phục vụ phân tích bổ sung; nhánh benchmark chính trong README này vẫn là **ba lớp** hoặc meta nhị phân.

---

## 7. Đặc trưng (feature engineering) — nhóm và ý nghĩa kinh tế

### 7.1. Core (giá–khối lượng–chỉ báo)

Gồm giá đóng, khối lượng, các biến **nến** (body, râu nến, biên độ so giá), gap open, các chỉ báo **đường trung bình động**, **RSI**, **MACD**, **Bollinger**, **OBV**, **VWAP**, và return/log-return. Đây là “ngôn ngữ” phân tích kỹ thuật cổ điển: động lượng, độ căng quá mua/quá bán, khối lượng tích lũy.

### 7.2. Regime (chế độ thị trường cục bộ)

Gồm lag return, tổng return nhiều phiên, **độ biến động rolling** (chuẩn độ lệch ngắn/dài), tỷ số biến động, **z-score** của biến động, khoảng cách giá so EMA. Ý tưởng: cùng một pattern giá nhưng trong **high-vol** vs **low-vol** có thể mang ý nghĩa khác nhau.

### 7.3. Market context (ngữ cảnh thị trường chéo)

Với mỗi ngày, pipeline gom nhóm toàn bộ mã trong panel và tạo đặc trưng kiểu:

- Return trung bình thị trường ngắn hạn và vài chu kỳ.
- **Breadth** (tỷ lệ mã tăng/giảm trong ngày).
- Thanh khoản tổng và biến động thanh khoản.
- **Độ mạnh tương đối** của mã so trung bình thị trường (relative strength).
- Phần khối lượng của mã so tổng thị trường.

Điều này giúp mô hình phân biệt “mã tăng vì cả rổ tăng” và “mã tăng khi rổ đi ngang”.

### 7.4. Seasonal / lịch giao dịch

Thứ trong tuần, tháng, quý, các biến sin/cos để tránh **ràng buộc tuyến tính cứng**, cờ cuối tháng/quý, vị trí phiên trong tháng/quý, tiến độ trong tháng/quý. Thị trường Việt Nam như nhiều nơi khác có **hiệu ứng lịch** (ví dụ dòng tiền cuối kỳ).

### 7.5. Hybrid econometric (không đưa vào ML input)

Các cột như dự báo return từ joint model, độ lệch chuẩn hóa, biến động GARCH, cờ hợp lệ… phục vụ **benchmark kinh tế lượng** và metadata, **không** trộn vào `feature_cols` của học máy (tránh leakage).

---

## 8. Meta-labeling kiểu AFML: EMA làm “primary”, triple-barrier làm nhãn phụ

### 8.1. Động cơ

Trong nhiều bài toán tài chính, ta có một **quy tắc đơn giản** (ví dụ: EMA nhanh cắt lên EMA chậm → ưu tiên long). Meta-labeling đặt câu hỏi thứ hai: **“Khi quy tắc phát tín hiệu, liệu kết cục có thường tốt không?”** — và để học máy học **xác suất** của câu trả lời đó trên các đặc trưng đã engineer.

### 8.2. Primary side

`primary_side = +1` nếu `EMA_10 > EMA_50`, ngược lại `-1`. Đây là tín hiệu **định hướng** đơn giản, dễ giải thích trong slide.

### 8.3. Triple-barrier trên đường giá (theo code)

Với mỗi mã, pipeline duyệt theo thời gian và với mỗi chỉ số \(i\) đủ dữ liệu tương lai:

- Chuẩn **độ biến động** cục bộ lấy từ **`VOL_20`** (độ lệch chuẩn return 20 phiên).
- **Chốt lời / cắt lỗ** được đặt **tỷ lệ** với \(\sigma\) hiện tại:  
  - Ngưỡng trên \(\approx pt \times \sigma_i\)  
  - Ngưỡng dưới \(\approx sl \times \sigma_i\)  
  (Tham số `meta_pt_sl` và `meta_barrier_horizon` trong `TargetConfig`.)

So với đường giá **chuẩn hóa** so entry \(i\), barrier đầu tiên chạm (lên/xuống) hoặc kết thúc chuỗi nếu không chạm trong horizon → gán `barrier_hit` và `barrier_path_ret`.

### 8.4. Gán `meta_label` nhị phân

Quy tắc (tóm tắt logic trong code):

- Nếu đang **long signal** (`primary_side > 0`): nhãn 1 nếu chạm **barrier trước** hoặc **đóng chuỗi có return dương** phù hợp ngữ cảnh; ngược lại 0.
- Nếu đang **short signal**: đối xứng với hướng giảm.

Sau đó các mô hình ML học **trên đặc trưng + nhãn meta** (binary). Khi suy diễn, xác suất “trade có ích” được so với **`meta_prob_trade_threshold`**; nếu quá thấp, pipeline có thể **ép phân bố xác suất về SIDEWAY** (tuỳ có đủ cột `primary_side` hay không — chi tiết trong `predict_trend_fusion`).

---

## 9. Chia train / validation / test theo thời gian

### 9.1. Vì sao không shuffle?

Chuỗi tài chính có **phụ thuộc thời gian**. Shuffle ngẫu nhiên sẽ **lạm dụng** mối tương quan giữa quá khứ và tương lai qua “rò rỉ” trộn nhãn tương lai vào đặc trưng quá khứ một cách không thực tế. Pipeline **sắp xếp theo `date`**, cắt:

- **Train:** đoạn đầu.  
- **Validation:** đoạn giữa (chiếm `val_ratio` của phần **trước test**).  
- **Test:** phần cuối (`test_ratio` của toàn bộ mẫu).

Đây là **walk-forward** đơn giản một lần cắt — đủ để báo cáo học thuật cơ bản; không thay thế **backtest giao dịch đầy đủ**.

### 9.2. Refit cho báo cáo test

Sau khi chọn feature set theo validation, mô hình được **fit lại trên train ∪ validation** rồi đánh giá trên **test** — đây là cách phổ biến để dùng hết dữ liệu huấn luyện cho mô hình cuối trước khi báo cáo độ chính xác out-of-sample chặt.

---

## 10. Chọn bộ đặc trưng và kiểm tra chất lượng dữ liệu

Với **mỗi** thuật toán và **mỗi** `feature_set` (`core`, `core_seasonal`):

1. **Giảm đa cộng tuyến** trên tập train giữa các cột tương quan cao.  
2. **Lọc theo permutation importance** so khả năng dự báo trên **validation**.  
3. **Kiểm tra dữ liệu** (`_run_ml_data_checks`): ví dụ ít nhất đủ mẫu mỗi lớp, không có vấn đề numerical cực đoan — nếu không pass thì **bỏ** cặp (model, feature set) đó.

Cuối cùng, với mỗi tên thuật toán, chọn bộ có **F1 weighted trên validation** cao nhất để báo cáo là **đại diện** của thuật toán đó trong bảng xếp hạng.

---

## 11. Các mô hình học máy — mô tả dài cho từng thuật toán

> **Lưu ý:** Bảng hyperparameter dưới đây là **mặc định trong code**. Nhánh **đa lớp** và nhánh **meta (nhị phân)** khác nhau ở Logistic / XGBoost / LightGBM như đã ghi.

### 11.1. Random Forest

**Intuition:** Tập hợp nhiều cây quyết định trên bootstrap mẫu và subset đặc trưng; trung bình biểu quyết hoặc xác suất lớp. Random Forest **ổn định**, ít cần co giãn đặc trưng, chịu nhiễu và ngoại lệ tốt hơn một cây đơn.

**Tham số chính trong repo:** `n_estimators=200`, `max_depth=10`, `class_weight="balanced"`, `n_jobs=2`, `random_state=42`.

**Vai trò trong hệ thống:** Thường đóng vai **baseline phi tuyến mạnh**; file `stock_model.joblib` lưu riêng một instance RF để tương thích đường cũ.

---

### 11.2. Logistic Regression (pipeline có `StandardScaler`)

**Intuition:** Mô hình tuyến tính trên không gian đã chuẩn hóa; ranh giới quyết định là siêu phẳng. Với **đa lớp**, dùng `multinomial` + `lbfgs`. Với **meta nhị phân**, để mặc định binary.

**Tham số:** `max_iter=2500`, `class_weight="balanced"`.

**Ý nghĩa báo cáo:** Cho thấy **mức trần** của mô hình tuyến tính — nếu gradient boosting vượt xa logistic một cách ổn định, có thể có **phi tuyến** quan trọng trong dữ liệu.

---

### 11.3. Histogram Gradient Boosting (HistGB)

**Intuition:** Boosting hiện đại của scikit-learn trên histogram của đặc trưng — nhanh, xử lý được tương tác và saturation.

**Tham số:** `max_iter=200`, `max_depth=6`, `learning_rate=0.05`.

---

### 11.4. Gradient Boosting cổ điển (GBT)

**Intuition:** Chuỗi cây yếu; mỗi bước sửa phần dư; `subsample` giúp regularization.

**Tham số:** `n_estimators=150`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`.

---

### 11.5. XGBoost

**Intuition:** Triển khai gradient boosting tối ưu hóa mạnh, phổ biến trong các cuộc thi dữ liệu dạng bảng.

**Đa lớp:** `objective="multi:softprob"`, `eval_metric="mlogloss"`.  
**Meta:** `objective="binary:logistic"`, `eval_metric="logloss"`.

**Điều kiện:** Chỉ có trong benchmark nếu import `xgboost` thành công.

---

### 11.6. LightGBM

**Intuition:** Cây phát triển theo lá (leaf-wise) với kiểm soát độ phức tạp qua `num_leaves`; thường hiệu quả và nhanh trên đặc trưng nhiều chiều.

**Đa lớp:** `objective="multiclass"`.  
**Meta:** `objective="binary"`.

**Điều kiện:** Cần `lightgbm`.

---

### 11.7. LSTM (Keras / TensorFlow)

**Intuition:** Mạng tái phân tán nhớ ngắn hạn trên **chuỗi cửa sổ** tạo từ vector đặc trưng mỗi ngày — nắm phụ thuộc cục bộ trong thời gian. Không phải mô hình “seq2seq giá thô” trong repo này; wrapper nhận **tabular** đã engineer.

**Tham số:** `units=16`, `dropout=0.1`, `epochs=20`, `batch_size=64`.

**Điều kiện:** Cần TensorFlow khi train và khi load bundle; môi trường thiếu TF có thể không khôi phục được trọng số LSTM (xem cảnh báo khi load).

---

## 12. Mô hình kinh tế lượng: AR(1) mean + GARCH(1,1) variance (joint)

### 12.1. Ý tưởng lý thuyết

Lợi suất có thể có **trung bình phụ thuộc chính nó** (AR) và **phương sai điều kiện thay đổi** (GARCH): biến động cụm sau các cơn sock.

### 12.2. Cách triển khai trong code

- Gói **`arch`**: mean **ARX với lag 1**, phương sai **GARCH(1,1)**, phân phối **chuẩn**.  
- Return được scale (`×100`) khi fit để ổn định số học; forecast mean và căn phương sai được **map ngược** về đơn vị return gốc.

### 12.3. Đánh giá relative với ML

Trên mỗi mã: fit trên **train window**, dự báo chuỗi **test window**, mapping forecast mean sang nhãn **TANG/SIDEWAY/GIAM** bằng cùng `_trend_label`, so với nhãn thực — ra accuracy/F1/precision weighted **theo encoder của ML**.

**Hàng “AR1-GARCH (joint)” trên bảng ranking:** trung bình metric qua các mã fit được — là **một đường đại diện** cho baseline time-series, không phải một model “uốn” theo đặc trưng ML.

---

## 13. Đánh giá: metric là gì và vì sao dùng

- **Accuracy:** Tỷ lệ dự đoán đúng trên test — dễ hiểu nhưng có thể **lạc quan** nếu một lớp chiếm đa số (SIDEWAY thường nhiều nếu ngưỡng nhỏ).

- **Precision / Recall / F1 weighted:** F1 weighted trung bình có trọng số theo support lớp — phản ánh tốt hơn khi **mất cân bằng**. Pipeline báo cáo precision/recall/F1 dạng weighted cho nhất quán.

- **ROC-AUC:** Khi có xác suất lớp và bài toán đủ điều kiện; với đa lớp có thể dùng loại **OvR weighted** (tuỳ phiên bản code).

---

## 14. Bảng xếp hạng, “best model”, và rolling-origin

### 14.1. Ranking

Sau khi có metric test cho **từng mô hình ML đã chọn feature set** và **một dòng econometric aggregate**, DataFrame được sort:

**`accuracy` giảm dần**, nếu trùng thì **`f1_weighted`** — điều này quan trọng khi trình bày: **đừng chỉ nhìn F1** vì có thể đảo thứ tự so với accuracy.

### 14.2. Rolling-origin

Nếu bật trong `TargetConfig`, pipeline có thể **đánh giá lăn** trên các ngày test: mỗi bước chỉ dùng quá khứ để train — gần với **pseudo-out-of-sample** hơn một lần cắt tĩnh. Chi tiết block JSON nằm trong `benchmark_summary["rolling_forecast_eval"]` khi có.

---

## 15. Suy diễn: fusion giữa ML và chuỗi thời gian

Hàm `predict_trend_fusion`:

1. Lấy **xác suất ML** từ mô hình được chọn (mặc định RandomForest nếu có).  
2. Nếu **meta-labeling**, điều chỉnh vector xác suất theo **meta probability** và **primary_side** / ngưỡng gate.  
3. Kết hợp với nhãn xu hướng từ **một bước** joint AR(1)–GARCH trên return quá khứ gần nhất (nếu `arch` khả dụng).  
4. Trọng số **`rolling_fusion_weight_ml`** nếu đã được tinh chỉnh và lưu trong bundle; nếu không có thì mặc định về phía ML mạnh hơn.

Fusion là **heuristic có kiểm soát** để báo cáo/demo — không phải chứng minh tối ưu lý thuyết Bayes.

---

## 16. File đầu ra, Colab, web, và tái lập môi trường

| Thành phần | Mô tả |
|------------|--------|
| `analysis_bundle.joblib` | Đối tượng Python chứa dict: models, benchmark, config, predictions store… |
| `stock_model.joblib` / `scaler.joblib` | Legacy tương thích (RF + metadata cột). |
| `artifacts/eda/` | Ảnh và JSON EDA sau `run_eda`. |
| `artifacts/colab_charts/` | Ảnh từ `bundle_chart_report.py`. |
| `colab.ipynb` | Hướng dẫn upload bundle + sinh chart trên Colab; **nên restart runtime** sau ô pip để tránh xung đột phiên bản thư viện. |

### 16.1. NumPy 1.x vs 2.x và joblib

Pickle/joblib nhạy với **phiên bản NumPy**. Hãy train và load bundle trên **cùng major** NumPy khi có thể. `requirements.txt` có gợi ý.

### 16.2. Web

`web_app.py` / `start_web.py` phục vụ dashboard nhẹ đọc bundle đã lưu — **khởi động lại server** sau khi train để thấy benchmark mới trong RAM.

---

## 17. Hạn chế khoa học và trách nhiệm sử dụng

1. **Hiệu năng quá khứ ≠ tương lai** — đặc biệt khi chế độ thị trường đổi (bull/bear/sideways).  
2. **Microstructure và chi phí** không được mô hình hóa đầy đủ.  
3. **Rò rỉ nhãn** đã được giảm bằng cách không đưa hybrid econometric vào ML features — nhưng vẫn có các nguồn leak khác tiềm ẩn (ví dụ lỗi pipeline nếu chỉnh tay sai).  
4. **Tính diễn giải ML phức tạp** — cần permutation importance / checklist trong benchmark để không “tin mù”.  
5. **Đạo đức:** không dùng kết quả để **khẳng định** lợi nhuận cho nhà đầu tư cá nhân mà không có khung pháp lý phù hợp.

---

## 18. Câu hỏi thường gặp (FAQ)

**Hỏi:** Vì sao có khi đủ 7 mô hình ML, có khi ít hơn?  
**Đáp:** Thiếu thư viện (XGB/LGB/TF), hoặc cặp (model, feature set) không pass kiểm tra / train lỗi → bị loại.

**Hỏi:** Meta-labeling có thay thế hoàn toàn nhãn đa lớp không?  
**Đáp:** Meta học nhị phân; khi suy diễn pipeline **ánh xạ** lại phân phối trên các lớp `TANG/SIDEWAY/GIAM` kết hợp `primary_side`, không phải “chỉ binary forever”.

**Hỏi:** Econometric có train chung weight với ML không?  
**Đáp:** Không — đường kinh tế lượng là benchmark **độc lập** trên return, aggregate sau.

**Hỏi:** Làm sao tái hiện đúng biểu đồ trong báo cáo?  
**Đáp:** Cùng commit code + cùng bundle + chạy `bundle_chart_report.py` với cùng phiên bản Python/NumPy/Pandas.

---

## 19. Phụ lục: lệnh chạy và cấu trúc thư mục

### 19.1. Lệnh cơ bản

```bash
pip install -r requirements.txt
python stock_analysis_yfinance.py       # train full pipeline + lưu bundle
python bundle_chart_report.py             # đọc bundle, xuất chart
python web_app.py                       # dashboard tại http://127.0.0.1:5000
```

### 19.2. Cấu trúc tối thiểu đáng nhớ

```
predictStockPrice/
├── stock_analysis_yfinance.py   # pipeline chính
├── bundle_chart_report.py         # sinh biểu đồ từ bundle
├── colab.ipynb                  # hướng dẫn Colab
├── requirements.txt
├── analysis_bundle.joblib         # sau khi train
├── stock_model.joblib
├── scaler.joblib
├── web_app.py / start_web.py      # (tuỳ chọn) giao diện web
├── templates/                     # HTML cho web
└── artifacts/
    ├── cache/                     # cache parquet (tạo khi chạy)
    ├── eda/                     # EDA (tạo khi chạy)
    └── colab_charts/              # chart báo cáo
```

---

*Nếu bạn chỉnh `TargetConfig` hoặc hyperparameter trong code, hãy cập nhật lại các con số trong tài liệu này khi nộp báo cáo chính thức — hoặc đính kèm tag git / hash commit để người đọc đối chiếu.*
