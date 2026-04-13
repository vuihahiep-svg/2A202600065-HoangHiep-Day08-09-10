# Group Report — Day 08 RAG Pipeline

## Team Goal
Nhóm triển khai một pipeline RAG hoàn chỉnh cho use case trợ lý nội bộ CS + IT Helpdesk, bao gồm 4 bước: indexing, retrieval, generation và evaluation. Mục tiêu chính là trả lời có nguồn trích dẫn, hạn chế hallucination, và có khả năng abstain khi thiếu bằng chứng.

## Technical Decisions
- Indexing dùng chunking theo heading (`=== Section ===`) kết hợp paragraph packing với `chunk_size=400`, `overlap=80`.
- Mỗi chunk giữ metadata chuẩn: `source`, `section`, `department`, `effective_date`, `access`.
- Vector store dùng ChromaDB local (`rag_lab` collection), embedding chạy ở chế độ offline bằng hash embedding 384 chiều để đảm bảo không phụ thuộc API ngoài.
- Baseline retrieval: dense top-10, chọn top-3 để build context.
- Variant tuning: hybrid (dense + BM25 bằng RRF) và bật rerank lexical nhẹ để giảm nhiễu.

## Sprint Outcomes
### Sprint 1
- Build index thành công cho đủ 5 tài liệu.
- Tổng chunk sau index: 29.
- Metadata coverage đầy đủ, không thiếu `effective_date`.

### Sprint 2
- `rag_answer("SLA xử lý ticket P1 là bao lâu?")` trả lời có citation `[1]`.
- `rag_answer("ERR-403-AUTH là lỗi gì?")` trả về abstain: "Không đủ dữ liệu...".
- Output có `sources` không rỗng với các câu có dữ liệu.

### Sprint 3
- Đã implement hybrid retrieval + rerank.
- So sánh A/B bằng scorecard cho thấy variant chưa vượt baseline trên tập test hiện tại.

### Sprint 4
- Chạy được full evaluation end-to-end.
- Đã sinh đủ artifacts:
  - `results/scorecard_baseline.md`
  - `results/scorecard_variant.md`
  - `results/ab_comparison.csv`

## Result Summary
Baseline:
- Faithfulness: 5.00
- Relevance: 4.20
- Context Recall: 5.00
- Completeness: 2.90

Variant:
- Faithfulness: 5.00
- Relevance: 3.90
- Context Recall: 5.00
- Completeness: 2.50

Delta variant - baseline:
- Faithfulness: +0.00
- Relevance: -0.30
- Context Recall: +0.00
- Completeness: -0.40

Kết luận nhóm: giữ baseline cho demo chính vì ổn định hơn với bộ test hiện tại.

## Lessons Learned
- Retrieval đúng source chưa đủ; cần đúng section để completeness không bị thấp.
- Cơ chế abstain giúp bảo toàn faithfulness, đặc biệt với câu thiếu dữ liệu (q09, q10).
- Offline mode giúp demo ổn định, nhưng muốn tăng completeness cần cải thiện answer planner (query rewrite + rule theo category).
