# Báo cáo tuần: kết quả Reproduce ViSpeechFormer

**Ngày:** 02/10/2026  
**Dataset:** ViVOS và LSVSC  

## 1. Việc đã làm

### 1.1. Tóm tắt công việc trong tuần

- **Chạy LSVSC lần đầu với config paper** (28/09). Transcript lấy từ mirror Hugging Face. Kết quả WER 21,11% (beam 5), gấp đôi paper. Bộ lọc loại tới 26,58% số câu, trong khi paper chỉ loại 9,24%. Vì vậy split test không còn tương đương với paper.
- **Thay transcript LSVSC bằng metadata chính thức.** Audio vẫn stream từ mirror `doof-ferb/LSVSC`, nhưng transcript được thay bằng ba file JSON chính thức. Sau khi thay, tỷ lệ câu bị lọc giảm xuống còn **5,87%**.
- **Áp dụng recipe sau khi improve cho LSVSC:** encoder 12 lớp, decoder 2 lớp, LFR(4,3), acoustic LayerNorm, Xavier, Noam, SpecAugment, label smoothing 0,1 và CTC ba luồng.

### 1.2. Kết quả trên LSVSC

Test có 5.367 câu sau lọc. Tất cả số liệu tính theo %, trong ngoặc là chênh lệch với paper.

| Model | Decode | CER ↓ | WER ↓ | PER-i ↓ | PER-r ↓ | PER-t ↓ | PER ↓ | OOV acc. ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ViSpeechFormer - paper | Không nêu rõ | 5,30 | 10,39 | 6,21 | 7,01 | 5,77 | 6,39 | **27,27** |
| Paper: Speech Transformer (char) | | 6,04 | 11,16 | | | | | |
| Paper: TASA (subword) | | 6,73 | 10,62 | | | | | |
| `best.pt` | Greedy | 4,77 | 8,84 | 5,19 | 6,00 | 4,60 | 5,31 | 20,93 |
| `best.pt` | Beam 5 | 4,49 | 8,28 | 4,91 | 5,66 | 4,36 | 5,01 | 22,48 |
| average top-5 | Greedy | 4,68 | 8,60 | 5,11 | 5,89 | 4,52 | 5,21 | 20,16 |
| average top-5 | Beam 5 | **4,35 (−0,95)** | **8,04 (−2,35)** | **4,77** | **5,49** | **4,21** | **4,86 (−1,53)** | 20,16 |

Các chỉ số phụ của run average beam 5:

| Chỉ số | Paper | average beam 5 |
|---|---:|---:|
| Số từ đúng phân biệt | 2.498 | 2.620 |
| Pearson / Spearman (tần suất train nhận đúng) | 21,54 / 29,54 | 21,48 / 26,32 |
| Latency trung bình (s/câu) | 0,1112 | 0,131 (greedy 0,114) |
| Lỗi thay thế / chèn / xoá | — | 6,03 / 1,23 / 0,78 |

### 1.3. Kết quả trên ViVOS

Test có 758 câu.

| Model | Decode | CER ↓ | WER ↓ | PER-i ↓ | PER-r ↓ | PER-t ↓ | PER ↓ | OOV đúng/tổng |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ViSpeechFormer - paper | Không nêu rõ | **11,96** | 30,49 | 12,18 | 20,52 | 13,39 | 15,42 | — |
| Paper: Speech Transformer (char) | | 18,54 | 34,83 | | | | | |
| Paper: Conv-Transformer (subword) | | 16,23 | 32,69 | | | | | |
| Reproduce lần 1 (config paper) | Beam 5 | 25,62 | 52,40 | 22,85 | 35,94 | 19,54 | 26,23 | 10/99 |
| `best.pt` (epoch 95) | Greedy | 15,65 | 34,08 | 12,92 | 22,48 | 13,14 | 16,24 | 13/99 |
| `best.pt` (epoch 95) | Beam 5 | 14,77 | 32,22 | 12,50 | 21,02 | 12,64 | 15,42 | 13/99 |
| average top-5 | Greedy | 14,24 | 31,73 | 11,44 | 20,81 | 12,01 | 14,81 | 15/99 |
| average top-5 | Beam 5 | 13,38 (+1,42) | **30,19 (−0,30)** | **11,07** | **19,42** | **11,44** | **14,04 (−1,38)** | **15/99** |

### 1.4. OOV accuracy

| Run | OOV đúng / tổng | OOV accuracy | KTC 95% |
|---|---:|---:|---:|
| Paper, LSVSC | 36 / 132 | 27,27% | 20,4–35,4% |
| LSVSC best, beam 5 | 29 / 129 | 22,48% | 16,1–30,4% |
| LSVSC best, greedy | 27 / 129 | 20,93% | 14,8–28,7% |
| LSVSC average, greedy / beam 5 | 26 / 129 | 20,16% | 14,1–27,9% |
| ViVOS average, greedy / beam 5 | 15 / 99 | 15,15% | 9,4–23,5% |
| ViVOS best, greedy / beam 5 | 13 / 99 | 13,13% | 7,8–21,2% |

Độ chính xác theo tần suất từ trong train, tính trên LSVSC average beam 5:

| Tần suất | OOV | 1–5 | 6–50 | 51–500 | > 500 |
|---|---:|---:|---:|---:|---:|
| Độ chính xác | 20,2% | 45,0% | 67,5% | 87,7% | 95,3% |

## 2. Phân tích kết quả

### 2.1. LSVSC: CER/WER/PER tốt hơn paper, nhưng split chưa tương đương

- **CER, WER và PER thấp hơn paper**, kể cả greedy decode trên `best.pt`. Kết quả tốt nhất đạt được dùng average + beam 5: WER 8,04%, thấp hơn paper 2,35 điểm, CER 4,35%, thấp hơn 0,95 điểm. Cả ba thành phần PER-i, PER-r và PER-t đều thấp hơn paper 1,4–1,6 điểm.
- **Chưa thể kết luận là đã vượt paper**, vì ba lý do:
  1. **Bộ lọc khác paper.** Paper loại 9,24% số câu, bản tái lập chỉ loại 5,87%. Test còn 5.367/5.683 câu, nên tập test không hoàn toàn trùng với paper. 
  2. **Chỉ có một seed.**
  3. **Recipe có các thành phần paper không mô tả:** LFR, Noam, SpecAugment, CTC phụ trợ và checkpoint averaging. Kết quả này cho thấy kiến trúc decoder âm vị đạt được mức lỗi paper báo cáo, nhưng không chứng minh được paper đã dùng đúng recipe này.
- **Mô hình chưa hội tụ hẳn.** Top-5 checkpoint theo dev WER đều nằm ở epoch 194–200. Dev WER vẫn đang giảm, nên train thêm có thể còn cải thiện.
- **Phân bố lỗi:** phần lớn là thay thế (6,03 điểm). Lỗi chèn (1,23) nhiều hơn lỗi xoá (0,78).

### 2.2. LSVSC: Kết quả trên OOV chưa đạt được

- OOV accuracy đạt 20,2–22,5%, thấp hơn paper 27,27% từ 4,8 đến 7,1 điểm. 
- WER chung tốt hơn paper nhưng OOV kém hơn. Điều này cho thấy phần cải thiện chủ yếu đến từ các từ tần suất cao. Độ chính xác tăng đơn điệu theo tần suất: từ 95,3% ở nhóm trên 500 lần xuống 20,2% ở nhóm OOV.
- Average làm OOV giảm nhẹ so với `best.pt`

### 2.3. ViVOS: WER ngang paper, CER còn cao hơn

- Average + beam 5 đạt **WER 30,19%, thấp hơn paper 0,30 điểm**, PER 14,04% cũng thấp hơn paper 1,38 điểm. Tuy vậy, **CER vẫn cao hơn 1,42 điểm**. Nghĩa là số từ sai tương đương paper, nhưng mỗi từ sai lệch nhiều ký tự hơn. Nguyên nhân chưa xác định, cần phân tích lỗi theo cặp ref/hyp.
- **Checkpoint averaging** giúp giảm WER 2,03 điểm (beam 5) so với `best.pt`

## 3. Kế hoạch tiếp theo

1. Ablation để biết cải thiện đến từ đâu
2. Thử nghiệm ý tưởng phoneme cho Vietnamses-English code-switching ASR
