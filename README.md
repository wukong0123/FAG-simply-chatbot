# FAQ Chatbot dùng Ollama và Chainlit

Ứng dụng hỏi đáp FAQ chạy cục bộ, tìm kiếm theo ngữ nghĩa bằng
`embeddinggemma` + NumPy và dùng `qwen3:1.7b` để diễn đạt câu trả lời. Chatbot
chỉ dùng nội dung FAQ được truy xuất và từ chối câu hỏi ngoài phạm vi.

## Kiến trúc

```text
data/faqs.json → ingest.py → Ollama Embedding
    → storage/*.npy/json → retriever.py
    → Ollama Chat → app.py (Chainlit)
```

Các module chính: `config.py` đọc cấu hình; `utils.py` validate dữ liệu và tính
cosine; `ollama_client.py` giao tiếp Ollama; `evaluate.py` đo chất lượng retrieval.

## Yêu cầu và cài đặt trên Windows

- Windows, Python 3.10 trở lên
- [Ollama](https://ollama.com/download) đang chạy

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
ollama pull embeddinggemma
ollama pull qwen3:1.7b
```

Nếu PowerShell chặn script kích hoạt, chạy
`Set-ExecutionPolicy -Scope Process Bypass` trong đúng cửa sổ terminal đó.

## Chạy ứng dụng

Tạo embedding sau lần cài đầu hoặc mỗi khi thay đổi `data/faqs.json`:

```powershell
python ingest.py
chainlit run app.py -w
```

Mở địa chỉ Chainlit hiển thị trong terminal (thường là
`http://localhost:8000`). Để chạy kiểm thử và đánh giá:

```powershell
pytest -q
python evaluate.py
```

Evaluation không gọi chat model nhưng vẫn cần Ollama embedding. Kết quả được
ghi vào `reports/test_results.csv` và `reports/evaluation_summary.json`.

## Cấu hình

Sửa `.env` để đổi URL, model, `TOP_K`, threshold, timeout hoặc đường dẫn dữ
liệu; không cần sửa source code. Đường dẫn tương đối được tính từ thư mục
project. `SIMILARITY_THRESHOLD` phải trong khoảng 0–1.

## Thay dữ liệu FAQ

Mỗi phần tử JSON cần `id`, `question`, `answer`; `category` không bắt buộc và
mặc định là `general`. ID phải duy nhất, chuỗi không được rỗng. Sau khi sửa:

```powershell
python ingest.py
```

Ứng dụng so sánh hash và sẽ yêu cầu ingest lại nếu dữ liệu đã đổi.

## Lỗi thường gặp

- Không kết nối Ollama: mở ứng dụng Ollama hoặc chạy `ollama serve`.
- Thiếu model: chạy `ollama pull embeddinggemma` và
  `ollama pull qwen3:1.7b`.
- Chưa có storage: chạy `python ingest.py`.
- HTTP 403: model cloud yêu cầu quyền/subscription; dùng model local mặc định.
- HTTP 429: đã chạm quota/rate limit, thử lại sau.
- Kết quả từ chối chưa phù hợp: hiệu chỉnh `SIMILARITY_THRESHOLD` bằng tập test.

## Giới hạn và hướng phát triển

MVP xử lý từng câu hỏi, chưa có đăng nhập, lịch sử dài hạn, vector database hay
reranker. Câu nhiều ý và FAQ gần nghĩa có thể cần hiệu chỉnh. Có thể phát triển
thêm hybrid search, reranking, FAISS/Qdrant, quản trị FAQ, Docker, logging/giám
sát và đánh giá bằng dữ liệu thực tế.
