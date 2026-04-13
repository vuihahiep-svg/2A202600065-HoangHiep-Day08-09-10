# Báo Cáo Cá Nhân — Lab Day 08: RAG Pipeline

**Họ và tên:** Nguyen Van A  
**Vai trò trong nhóm:** Retrieval Owner  
**Ngày nộp:** 2026-04-13  
**Độ dài:** ~640 từ

---

## 1. Tôi đã làm gì trong lab này?

Trong lab này, tôi tập trung chính vào phần Retrieval Owner, cụ thể là các hạng mục liên quan đến Sprint 1, Sprint 2 và Sprint 3. Ở Sprint 1, tôi xử lý pipeline index: đọc 5 tài liệu trong `data/docs`, parse metadata từ header (source, department, effective_date, access), sau đó chunk theo heading và paragraph để giữ ngữ cảnh tự nhiên của điều khoản. Tôi thiết kế chunking theo chiến lược heading-first để tránh cắt giữa các mục như “Điều 2”, “Section 4”, hoặc các bullet quan trọng trong SLA.

Ở Sprint 2, tôi implement dense retrieval qua ChromaDB và chuẩn hóa output chunk gồm text + metadata + score. Tôi cũng tham gia chỉnh flow trả lời theo hướng grounded, trong đó câu trả lời luôn phải dựa trên context retrieve được và có citation.

Ở Sprint 3, tôi implement thêm hybrid retrieval (dense + sparse BM25) và rerank lexical nhẹ để thử giảm noise. Phần của tôi kết nối trực tiếp với phần Eval Owner vì scorecard của Sprint 4 phụ thuộc hoàn toàn vào chất lượng retrieve/selection.

---

## 2. Điều tôi hiểu rõ hơn sau lab này

Điều tôi hiểu rõ nhất sau lab là chunking và retrieval thực sự quyết định chất lượng RAG nhiều hơn tôi tưởng. Trước đây tôi thường nghĩ chỉ cần embedding tốt là đủ, nhưng khi làm thực tế, tôi thấy nếu chunk không đúng ranh giới logic thì model vẫn có thể trả lời sai dù source đúng.

Concept thứ hai tôi hiểu sâu hơn là “retrieve đúng source chưa đủ, phải retrieve đúng section”. Ví dụ tài liệu `access_control_sop` có nhiều section đều nói về “quyền”, “phê duyệt”, “escalation”, nhưng câu hỏi “Level 3 cần ai phê duyệt” chỉ đúng ở một đoạn rất cụ thể. Nếu retrieve nhầm sang phần escalation thì answer vẫn nghe hợp lý nhưng thiếu ý chính.

Tôi cũng hiểu rõ giá trị của abstain rule. Trong lab này, có câu kiểu ERR-403-AUTH không có đủ thông tin trong docs; nếu không có cơ chế abstain, hệ thống rất dễ bịa. Khi ép abstain rõ ràng, faithfulness giữ ổn định và tránh penalty nặng.

---

## 3. Điều tôi ngạc nhiên hoặc gặp khó khăn

Khó khăn lớn nhất của tôi là tuning retrieval khi chạy offline, không dùng external API embeddings/LLM. Điều này làm tôi phải ưu tiên các heuristic và lexical signals nhiều hơn, trong khi vẫn giữ luồng RAG chuẩn. Ban đầu tôi nghĩ hybrid + rerank chắc chắn sẽ tốt hơn baseline dense, nhưng kết quả scorecard cho thấy không phải lúc nào cũng vậy.

Một điểm gây mất thời gian debug là hiện tượng “đúng tài liệu, sai đoạn”. Ví dụ câu về escalation P1 hoặc alias “Approval Matrix”, hệ thống retrieve được tài liệu access hoặc SLA đúng domain nhưng chọn câu không trực tiếp trả lời expected answer. Hypothesis ban đầu của tôi là do top-k quá thấp, nhưng khi tăng search không giải quyết triệt để. Thực tế vấn đề nằm ở bước select/rerank chưa ưu tiên đủ mạnh vào intent chính của query.

Ngoài ra, tôi ngạc nhiên vì completeness khó tối ưu hơn faithfulness. Faithfulness có thể cao nhờ grounded + abstain, nhưng completeness cần extract đúng các chi tiết cụ thể (số phút, tên vai trò phê duyệt, điều kiện ngoại lệ). Đây là bài học rõ ràng cho thiết kế production RAG.

---

## 4. Phân tích một câu hỏi trong scorecard

**Câu hỏi:** q07 — “Approval Matrix để cấp quyền hệ thống là tài liệu nào?”

Ở baseline dense, hệ thống thường retrieve đúng tài liệu `it/access-control-sop.md`, nhưng câu trả lời chưa nêu trực diện mapping alias “Approval Matrix” sang tên tài liệu mới “Access Control SOP”. Vì vậy completeness của baseline ở câu này vẫn thấp (1/5), dù faithfulness cao vì không bịa ngoài context. Nói cách khác, retrieval có recall tốt (expected source vẫn có) nhưng generation chưa “resolve alias” thành câu trả lời chính xác theo expected answer.

Với variant hybrid + rerank, tôi kỳ vọng BM25 sẽ giúp bắt từ khóa “Approval Matrix” tốt hơn. Tuy nhiên, kết quả lại cho thấy variant hay abstain hơn ở câu này và relevance giảm nhẹ. Theo tôi, nguyên nhân là ở chế độ offline, heuristic rerank của nhóm còn đơn giản, chưa đủ mạnh để đẩy đúng chunk alias lên top khi cạnh tranh với các chunk có nhiều từ “quyền hệ thống” nói chung.

Root cause chính ở đây là retrieval/selection, không phải indexing. Vì metadata và source đã đầy đủ, vấn đề nằm ở cách chọn chunk cuối cùng cho prompt. Nếu có thêm thời gian, tôi sẽ thêm rule query rewrite cho alias (“Approval Matrix” -> “Access Control SOP”) trước khi retrieve để tăng precision.

---

## 5. Nếu có thêm thời gian, tôi sẽ làm gì?

Nếu có thêm thời gian, tôi sẽ thử hai cải tiến cụ thể. Thứ nhất, thêm query rewriting theo dictionary alias dựa trên lỗi q07 để map tên cũ sang tên mới trước retrieval. Thứ hai, tôi sẽ tách answer strategy theo category (SLA, Access, Refund) để tăng completeness ở các câu cần số liệu/điều kiện cụ thể như q06 và q10. Hai cải tiến này trực tiếp dựa trên scorecard vì baseline đang mạnh ở faithfulness nhưng yếu ở completeness.
