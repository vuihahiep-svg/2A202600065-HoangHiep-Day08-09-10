# Tuning Log — RAG Pipeline (Day 08 Lab)

> Template: Ghi lại mỗi thay đổi và kết quả quan sát được.
> A/B Rule: Chỉ đổi MỘT biến mỗi lần.

---

## Baseline (Sprint 2)

**Ngày:** 2026-04-13  
**Config:**
```
retrieval_mode = "dense"
chunk_size = 400 tokens
overlap = 80 tokens
top_k_search = 10
top_k_select = 3
use_rerank = False
llm_model = offline_extractive
```

**Scorecard Baseline:**
| Metric | Average Score |
|--------|--------------|
| Faithfulness | 5.00 /5 |
| Answer Relevance | 4.20 /5 |
| Context Recall | 5.00 /5 |
| Completeness | 2.90 /5 |

**Câu hỏi yếu nhất (điểm thấp):**
- `q06` (Escalation P1): completeness thấp vì answer chưa lấy đúng dòng "auto escalate lên Senior Engineer trong 10 phút".
- `q07` (Approval Matrix alias): answer chưa map rõ "Approval Matrix" -> "Access Control SOP", completeness thấp.
- `q10` (VIP refund): hệ thống abstain đúng hướng anti-hallucination nhưng thiếu diễn giải theo policy hiện hành.

**Giả thuyết nguyên nhân (Error Tree):**
- [ ] Indexing: Chunking cắt giữa điều khoản
- [ ] Indexing: Metadata thiếu effective_date
- [x] Retrieval: Dense bỏ lỡ exact keyword / alias
- [ ] Retrieval: Top-k quá ít → thiếu evidence
- [x] Generation: Prompt không đủ grounding
- [ ] Generation: Context quá dài → lost in the middle

---

## Variant 1 (Sprint 3)

**Ngày:** 2026-04-13  
**Biến thay đổi:** Retrieval mode + Rerank (`dense` -> `hybrid` + `use_rerank=True`)  
**Lý do chọn biến này:**
Baseline đã retrieve đúng source nhưng một số câu vẫn trả lời lệch section (q06, q07).  
Giả thuyết: dense-only chưa đủ mạnh cho keyword alias; hybrid (dense + sparse) + rerank có thể lọc tốt hơn phần keyword và giảm nhiễu.

**Config thay đổi:**
```
retrieval_mode = "hybrid"   # hoặc biến khác
top_k_search = 10
top_k_select = 3
use_rerank = True
# Các tham số còn lại giữ nguyên baseline
```

**Scorecard Variant 1:**
| Metric | Baseline | Variant 1 | Delta |
|--------|----------|-----------|-------|
| Faithfulness | 5.00/5 | 5.00/5 | +0.00 |
| Answer Relevance | 4.20/5 | 3.90/5 | -0.30 |
| Context Recall | 5.00/5 | 5.00/5 | +0.00 |
| Completeness | 2.90/5 | 2.50/5 | -0.40 |

**Nhận xét:**
Variant giữ tốt anti-hallucination (faithfulness 5.0 ổn định) nhưng kém baseline ở relevance/completeness.  
Các câu q01, q06, q07 có xu hướng abstain nhiều hơn nên giảm completeness.  
Không có cải thiện rõ rệt trên bộ test hiện tại, chứng tỏ biến tuning này chưa phù hợp với heuristic generator offline.

**Kết luận:**
Variant 1 **không tốt hơn** baseline trên test set nội bộ.  
Bằng chứng: Delta âm ở relevance (-0.30) và completeness (-0.40), trong khi faithfulness/recall giữ nguyên.  
Quyết định cuối: giữ cấu hình baseline dense cho demo chính.

---

## Variant 2 (nếu có thời gian)

**Biến thay đổi:** ___________  
**Config:**
```
# TODO
```

**Scorecard Variant 2:**
| Metric | Baseline | Variant 1 | Variant 2 | Best |
|--------|----------|-----------|-----------|------|
| Faithfulness | ? | ? | ? | ? |
| Answer Relevance | ? | ? | ? | ? |
| Context Recall | ? | ? | ? | ? |
| Completeness | ? | ? | ? | ? |

---

## Tóm tắt học được

1. **Lỗi phổ biến nhất trong pipeline này là gì?**
   > Retrieval đúng tài liệu nhưng sai đoạn, khiến generation trả lời thiếu ý hoặc lệch trọng tâm.

2. **Biến nào có tác động lớn nhất tới chất lượng?**
   > Chất lượng chọn chunk cuối cùng (selection/rerank) tác động mạnh hơn việc tăng số chunk retrieve.

3. **Nếu có thêm 1 giờ, nhóm sẽ thử gì tiếp theo?**
   > Thử query rewrite riêng cho alias (Approval Matrix -> Access Control SOP) và tách rule answer theo từng category (SLA, access, refund).
