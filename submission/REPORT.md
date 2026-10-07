# Lab 21 — Evaluation Report

**Họ tên**: Võ Đức Trí  **MSSV**: 2A202602603  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 14.6GB (sm_75, fp16)`

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được là 98 *(results/token_stats.json)*, mức gợi ý 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** Có (`reasoning preserved — safe to train on traces` trong `results/template_check.json`).  
Qwen3.5 giữ nguyên thẻ `<think>` và đóng thẻ rỗng trong lượt prompt sinh, không làm mất dấu suy luận và dữ liệu đầu ra được bảo toàn trọn vẹn.

---

## 2. Mask proof (NB1)

| Tiêu chí | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49%) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3142.5 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 972.2 |
| (c) LoRA fine-tune | 0.965 | 0.544 | 1.000 | 1344.1 |

**(b) có thật sự mạnh hơn (a) không?** Có, (b) đạt target 0.765 và format 1.000 trong khi (a) hoàn toàn thất bại ở format (0.000) và target (0.000). Độ trễ của (b) cũng nhanh hơn gấp 3.2 lần (972ms so với 3142.5ms) do output JSON ngắn gọn, không bị lặp từ hay giải thích lan man.  
Tôi không chỉnh sửa `OPTIMIZED_PROMPT` nhằm giữ nguyên tính khách quan và đảm bảo SHA đối chiếu (`719e74d3b6232053`) hoàn toàn khớp với mốc đóng băng ban đầu của lab trên toàn bộ 50 mẫu đánh giá.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 0.0001 | 0.6281 | **0.9650** | 377.7 | 8.78 |
| `attn_only` | q,v | 283 | 32,456,704 | 0.0001 | **0.5371** | **0.9700** | 254.8 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | **0.0000** | 378.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | **0.9400** | 447.6 | **3.86** |

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập target 50 mẫu, `attn_only` đạt 0.9700, xấp xỉ tương đương với `correct` (0.9650, chênh lệch chỉ 1 mẫu). Tuy nhiên, nếu nhìn vào cột train loss ở NB4, `attn_only` lại có loss thấp hơn hẳn (0.5371 so với 0.6281). Việc ép rank cực lớn ($r=283$) vào chỉ 2 ma trận chiếu $q, v$ giúp adapter tối ưu hóa hàm loss huấn luyện mạnh mẽ hơn (overfitting loss do tập trung tham số cục bộ), nhưng năng lực suy diễn trên tập kiểm thử thực tế không tạo ra cách biệt vượt trội so với cấu hình `correct` ($r=16$) dàn trải trên toàn bộ 12 khối text-linear. Điều này khẳng định kết luận quan trọng của deck: việc so sánh công bằng phải thực hiện ở cấp độ ngân sách tham số (matched budget) và đánh giá trên năng lực tác vụ cuối cùng chứ không thể dựa vào train loss để xếp hạng.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
`wrong_lr` sử dụng learning rate cấp độ full fine-tuning ($10^{-5}$ thay vì $10^{-4}$ cho LoRA), khiến tốc độ học gần như bị tê liệt hoàn toàn: loss chỉ giảm từ 2.163 xuống 1.119 ở epoch cuối (train loss trung bình 1.5702, cao gấp 2.5 lần `correct`). Khi đánh giá trên tác vụ target, `wrong_lr` hoàn toàn thất bại với target = 0.0000 và format = 0.0000, độ trễ sinh từ vọt lên 5070ms do mô hình rơi vào vòng lặp suy diễn vô tận. Nếu chỉ nhìn đường loss đi ngang mà không biết nguyên nhân do LR quá nhỏ, người làm thí nghiệm rất dễ kết luận sai lầm rằng dữ liệu bị lỗi, kiến trúc LoRA không học được tác vụ JSON triage, hoặc cần phải tăng thêm tham số mô hình.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
`qlora` tiết kiệm VRAM rất ấn tượng: giảm từ 8.78 GB xuống chỉ còn 3.86 GB (tiết kiệm hơn 56% bộ nhớ đồ họa). Tuy nhiên, cái giá phải trả là thời gian huấn luyện tăng lên đáng kể (447.6s so với 377.7s, chậm hơn khoảng 18.5% do overhead giải nén trọng số lượng tử hóa 4-bit liên tục trên kiến trúc Turing T4), và độ chính xác target bị suy giảm từ 0.9650 xuống 0.9400. Kết quả thực nghiệm này hoàn toàn ủng hộ khuyến nghị từ deck (§13): khi phần cứng T4 14.6GB vẫn đủ khả năng chạy 16-bit LoRA (8.78 GB VRAM), việc sử dụng 4-bit QLoRA cho Qwen3.5 là không tối ưu vì làm tụt hiệu năng tác vụ và kéo dài thời gian huấn luyện.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.200` · `regression Δ = -0.247` · `valid_trace_rate = 0.00`

**Diễn giải:**  
Phán quyết của cổng kiểm định là FAILED vì độ chính xác trên tập năng lực tổng quát (`regression`) bị tụt 0.247 (từ 0.791 xuống 0.544), vượt quá ngưỡng dung sai cho phép là 0.020. Mặc dù ở tác vụ mục tiêu CSKH, bản fine-tune LoRA đã thắng áp đảo baseline prompt tối ưu (target tăng từ 0.765 lên 0.965, tức $\Delta = +0.200$), mô hình đã gặp phải hiện tượng quên thảm họa (catastrophic forgetting). Toàn bộ 225 mẫu huấn luyện chỉ thuần túy là dữ liệu ticket CSKH dạng JSON có cấu trúc hẹp, khiến mô hình bị lệch phân phối và suy giảm khả năng trả lời các câu hỏi kiến thức thông thường. Để vượt qua cổng hồi quy này theo lý thuyết deck §6.3, giải pháp chuẩn mực là bổ sung 1–5% dữ liệu tổng quát (replay data) vào tập huấn luyện để duy trì năng lực nền tảng của base model.

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. | hoan_tien, cao, bình giữ nhiệt, tieu_cuc | hoan_tien, cao, bình giữ nhiệt, tieu_cuc | hoan_tien, trung_binh, bình giữ nhiệt, trung_tinh | ❌ **FT thua** (FT đoán hạ mức urgency từ cao xuống trung bình) |
| 2 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. | san_pham_loi, cao, nồi chiên không dầu, tieu_cuc | san_pham_loi, cao, nồi chiên không dầu, tieu_cuc | san_pham_loi, trung_binh, nồi chiên không dầu, trung_tinh | ❌ **FT thua** (FT đoán hạ mức urgency từ cao xuống trung bình) |
| 3 | Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. | san_pham_loi, thap, áo khoác gió, trung_tinh | san_pham_loi, thap, áo khoác gió, trung_tinh | san_pham_loi, trung_binh, áo khoác gió, trung_tinh | ❌ **FT thua** (FT đoán nhầm urgency từ thấp thành trung bình) |
| 4 | Cho mình hỏi, mình đặt ốp lưng điện thoại mã đơn DH936478. Shipper khô | van_chuyen, thap, ốp lưng điện thoại, tieu_cuc | van_chuyen, trung_binh, ốp lưng điện thoại, tieu_cuc | van_chuyen, thap, ốp lưng điện thoại, tieu_cuc | ✅ **FT thắng** (FT nhận diện chính xác mức độ khẩn cấp thấp) |
| 5 | Alo shop, mình đặt ốp lưng điện thoại mã đơn DH734695. Giá bao nhiêu. | hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh | hoi_thong_tin, thap, ốp lưng điện thoại, trung_tinh | hoi_thong_tin, trung_binh, ốp lưng điện thoại, trung_tinh | ✅ **FT thắng** (FT xác định đúng mức khẩn cấp trung bình) |

**Có mẫu chung nào ở các ca FT thua không?**  
Các ca FT bị mất điểm tập trung chủ yếu ở trường `urgency` và `sentiment`. Khi khách hàng dùng từ ngữ ngắn gọn (như "Chưa thấy tiền", "Thiếu phụ kiện"), bản prompt (b) nhờ có phần mô tả ngữ cảnh chi tiết trong system prompt nên đã nhận diện chính xác mức độ khẩn cấp là `cao`. Trong khi đó, bản fine-tune LoRA có xu hướng thiên kiến về nhãn phổ biến `trung_binh`, cho thấy tập huấn luyện 225 mẫu còn mất cân bằng nhãn phân loại độ khẩn cấp.

---

## 7. Kết luận & điều tôi học được

**Kết luận:**  
Từ kết quả thực nghiệm trên GPU Tesla T4 với tập dữ liệu kiểm thử đầy đủ 50 mẫu, câu trả lời cho việc có nên triển khai bản fine-tune này vào môi trường sản xuất hay không là: **Chưa nên triển khai ngay nếu hệ thống phục vụ đa tác vụ, nhưng hoàn toàn có thể triển khai nếu tách biệt thành microservice chuyên biệt cho triage CSKH.**  
Về mặt tích cực, bản LoRA `correct` đã vượt trội mốc baseline (b) ở bài toán nghiệp vụ chính (target 0.965 so với 0.765), đảm bảo chuẩn định dạng JSON 100% và rút ngắn đáng kể chiều dài prompt đầu vào. Tuy nhiên, việc năng lực tổng quát bị suy giảm (-0.247 ở bài kiểm tra hồi quy) cảnh báo nguy cơ mô hình phản hồi sai lệch nếu người dùng gửi các câu hỏi nằm ngoài phạm vi CSKH. Đòn bẩy thực sự trong lab này không nằm ở rank hay thủ thuật lượng tử hóa, mà nằm ở **loss mask chuẩn xác (NB1)** bảo đảm mô hình học câu trả lời thay vì prompt, **vị trí gán adapter (all-linear)** giúp biểu diễn tri thức đồng đều, và **learning rate đúng thang độ LoRA ($10^{-4}$)** để kích hoạt khả năng thích nghi của mô hình.

**Ba điều tôi học được:**
1. **Kiểm tra Loss Mask trước khi tốn thời gian train:** Giải mã ngược token ở vị trí `labels != -100` là bước kiểm tra sống còn; nếu tính loss trên cả prompt, toàn bộ quá trình huấn luyện sẽ trở thành công cốc.
2. **Train loss là chỉ số thay thế dễ gây ngộ nhận:** Adapter `attn_only` dù có train loss thấp hơn `correct` nhờ rank cao ghi nhớ cục bộ, nhưng ở năng lực tác vụ thực tế lại không vượt trội hơn; việc tối ưu hóa đúng vị trí linear quan trọng hơn việc dồn ép tham số vào một vài ma trận attention.
3. **Mốc so sánh Baseline (b) phải được đóng băng nghiêm ngặt:** Nếu không thiết lập một đối thủ mạnh (few-shot tối ưu) trước khi huấn luyện, chúng ta sẽ rất dễ tự huyễn hoặc về giá trị thực sự của việc fine-tune.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:**  
Trộn thêm 3% dữ liệu hội thoại tổng quát (replay buffer) vào tập 225 mẫu huấn luyện để giải quyết dứt điểm hiện tượng suy thoái năng lực chung, từ đó vượt qua cổng hồi quy NB5 với kết quả `PASSED`.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
