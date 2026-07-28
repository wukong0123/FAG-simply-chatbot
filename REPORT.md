# Báo cáo FAQ Chatbot

## 1. Mục tiêu

Nghiên cứu và xây dựng chatbot sử dụng LLM, trả lời dựa trên FAQ và nhận diện
các cách diễn đạt tương đồng về mặt ngữ nghĩa, đồng thời không bịa thông tin.

## 2. Giải pháp đã khảo sát

- Open WebUI: giao diện hoàn chỉnh nhưng phần luồng retrieval khó quan sát hơn.
- AnythingLLM: nhiều tính năng quản trị, vượt nhu cầu MVP nhỏ.
- Chainlit + Ollama + semantic retrieval: ít thành phần, chạy local, source dễ
  đọc và các bước có thể kiểm thử độc lập.

Giải pháp thứ ba được chọn để sinh viên có thể theo dõi toàn bộ quá trình và
thay đổi dữ liệu/model mà không sửa kiến trúc.

## 3. Kiến trúc hệ thống

```text
FAQ → Embedding → Similarity Search → Context → LLM → Chainlit
```

Embedding và metadata lưu trong file cục bộ. Retriever tính cosine similarity
bằng NumPy, lọc threshold rồi mới gọi LLM.

## 4. Công nghệ sử dụng

Python, Chainlit, Ollama, EmbeddingGemma, Qwen3, NumPy và Pytest.

## 5. Luồng xử lý

`ingest.py` validate FAQ, tạo văn bản embedding, gọi Ollama, chuẩn hóa vector và
lưu ma trận cùng hash nội dung. Với mỗi câu hỏi, `retriever.py` tạo query
embedding, xếp hạng top-k và từ chối nếu score cao nhất dưới threshold.
`ollama_client.py` chỉ gửi các FAQ phù hợp làm context. `app.py` hiển thị câu
trả lời và nguồn; nếu chat model lỗi, câu trả lời FAQ gần nhất được dùng.

## 6. Kết quả

Điền số liệu sau khi chạy `python evaluate.py`:

- Tổng số test: _(cập nhật từ `evaluation_summary.json`)_
- Hit@1: _(cập nhật)_
- Hit@3: _(cập nhật)_
- Reject accuracy: _(cập nhật)_
- Latency trung bình: _(cập nhật)_
- Ảnh giao diện: _(chèn ảnh sau khi chạy Chainlit)_
- Ví dụ paraphrase: “Công ty mấy giờ mở cửa?”
- Ví dụ ngoài phạm vi: “Hôm nay thời tiết có mưa không?”

## 7. Khó khăn

Ollama chưa chạy gây ConnectionRefused; model cloud có thể yêu cầu subscription;
model nhỏ đôi khi diễn đạt chưa tự nhiên; threshold cần hiệu chỉnh; câu nhiều ý
khó truy xuất hơn câu một ý.

## 8. Giới hạn

Dữ liệu FAQ còn nhỏ, chưa có vector database, authentication hay lịch sử dài
hạn. Multi-intent chưa được tách ý. Chất lượng phụ thuộc embedding và chat model.

## 9. Hướng phát triển

Thử Chroma/FAISS/Qdrant, reranking, hybrid search, màn hình quản trị FAQ,
Docker, trích dẫn chi tiết, monitoring, model tốt hơn và tập đánh giá thực tế.
