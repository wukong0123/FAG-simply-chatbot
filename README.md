# FAQ Chatbot — Ollama Cloud + Gemini Embedding

Chatbot FAQ dùng `gpt-oss:20b` trên Ollama Cloud để sinh câu trả lời,
`gemini-embedding-001` để tìm kiếm ngữ nghĩa và NumPy để xếp hạng. Hệ thống
chỉ trả lời dựa trên FAQ được truy xuất, hiển thị nguồn và từ chối câu hỏi
ngoài phạm vi.

## Kiến trúc

```text
data/faqs.json
    → ingest.py
    → Gemini Embedding API
    → storage/embeddings.npy + metadata.json

Câu hỏi người dùng
    → Gemini query embedding
    → NumPy cosine similarity
    → top-k FAQ + threshold
    → trả trực tiếp FAQ hoặc gọi Ollama Cloud
    → Chainlit UI
```

Các module chính:

- `config.py`: đọc và kiểm tra biến môi trường.
- `embedding_client.py`: Gemini Cloud hoặc Ollama local embedding.
- `ollama_client.py`: gọi Ollama Cloud để sinh câu trả lời.
- `ingest.py`: tạo embedding cho câu hỏi chính và từng alias.
- `retriever.py`: semantic search, top-k và loại kết quả trùng FAQ.
- `app.py`: giao diện Chainlit và bộ nhớ hội thoại ngắn hạn.
- `evaluate.py`: đánh giá Hit@1, Hit@3, reject accuracy và latency.

## Yêu cầu

- Windows và Python 3.10 trở lên.
- Ollama API key tại <https://ollama.com>.
- Gemini API key tại <https://aistudio.google.com/apikey>.

Không cần cài Ollama hoặc tải model xuống máy khi dùng cấu hình cloud mặc định.

## Cài đặt

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Nếu PowerShell chặn script kích hoạt:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Cấu hình cloud

Điền API key vào `.env`:

```env
OLLAMA_BASE_URL=https://ollama.com
OLLAMA_API_KEY=
OLLAMA_CHAT_MODEL=gpt-oss:20b

EMBEDDING_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768

TOP_K=3
SIMILARITY_THRESHOLD=0.55
FAQ_DATA_PATH=data/faqs.json
EMBEDDINGS_PATH=storage/embeddings.npy
METADATA_PATH=storage/metadata.json
REQUEST_TIMEOUT_SECONDS=120
```

Không gửi API key xuống frontend và không commit `.env`. File này đã được
khai báo trong `.gitignore`.

## Chạy ứng dụng

Chạy ingest sau lần cài đầu, khi thay đổi FAQ hoặc khi đổi embedding
provider/model:

```powershell
python ingest.py
chainlit run app.py -w
```

Mở <http://localhost:8000>.

## Kiểm thử và đánh giá

```powershell
pytest -q
python evaluate.py
```

Evaluation gọi Gemini Embedding cho mỗi test nhưng không bắt buộc gọi chat
model. Kết quả được ghi vào:

- `reports/test_results.csv`
- `reports/evaluation_summary.json`

## Dữ liệu FAQ và alias

Mỗi FAQ có schema:

```json
{
  "id": 1,
  "question": "Câu hỏi chính",
  "aliases": ["Cách hỏi tương tự", "Cách diễn đạt khác"],
  "answer": "Câu trả lời",
  "category": "category_name"
}
```

`id`, `question` và `answer` là bắt buộc. `category` mặc định là `general`;
`aliases` là danh sách tùy chọn. Mỗi câu hỏi chính và alias có embedding riêng,
sau đó retriever gộp các kết quả trùng FAQ.

Sau khi sửa dữ liệu:

```powershell
python ingest.py
```

Ứng dụng dùng hash để phát hiện dữ liệu hoặc embedding model đã thay đổi.

## Bộ nhớ hội thoại

Mỗi phiên Chainlit ghi nhớ tối đa 8 tin nhắn gần nhất trong
`cl.user_session`. Lịch sử được gửi kèm khi gọi Ollama Cloud:

- Mỗi phiên trình duyệt có lịch sử riêng.
- Restart server hoặc tạo phiên mới sẽ mất lịch sử.
- Chưa hỗ trợ chia sẻ lịch sử giữa nhiều backend instance.

Khi triển khai production nhiều người dùng, nên thay bộ nhớ RAM bằng Redis,
dùng khóa `chat:{user_id}:{conversation_id}` và TTL phù hợp.

## Cách hệ thống chọn câu trả lời

- Top-1 dưới threshold: trả fallback và không gọi chat model.
- Top-1 rõ ràng: trả trực tiếp đáp án FAQ để giảm latency và hallucination.
- Kết quả mơ hồ: gửi FAQ liên quan và lịch sử phiên cho Ollama Cloud.
- Chat model lỗi nhưng retrieval tốt: fallback về đáp án FAQ gần nhất.

## Giới hạn cloud

`gpt-oss:20b` được chọn vì tài khoản Ollama Free có thể sử dụng model nhẹ này.
Ollama tính usage theo thời gian GPU, độ dài prompt và độ dài câu trả lời,
không công bố số câu cố định. Gói Free chỉ phù hợp demo hoặc nhóm nhỏ; ứng
dụng công khai cần rate limit, cache và gói cloud phù hợp.

Gemini Embedding có quota theo project và usage tier. Khi gặp HTTP 429, giảm
tần suất request hoặc kiểm tra quota trong Google AI Studio.

## Chuyển về Ollama local

Đổi `.env`:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_API_KEY=
OLLAMA_CHAT_MODEL=qwen3:1.7b
EMBEDDING_PROVIDER=ollama
OLLAMA_EMBEDDING_MODEL=embeddinggemma
```

Tải model và ingest lại:

```powershell
ollama pull embeddinggemma
ollama pull qwen3:1.7b
python ingest.py
```

## Lỗi thường gặp

- Chainlit báo `Could not reach the server` sau khi gửi câu hỏi: các lời gọi
  Gemini/Ollama phải chạy bất đồng bộ để WebSocket vẫn duy trì heartbeat.
  Phiên bản hiện tại đã xử lý việc này và sẽ hiển thị lỗi timeout nếu dịch vụ
  cloud không phản hồi trong `REQUEST_TIMEOUT_SECONDS`.
- Truy cập từ máy khác trong mạng LAN: chạy
  `chainlit run app.py --host 0.0.0.0 --port 8000`, mở cổng 8000 trên firewall
  và truy cập bằng địa chỉ IP của máy chạy chatbot. `127.0.0.1` chỉ cho phép
  truy cập ngay trên chính máy đó.
- Ollama 401/403: kiểm tra API key hoặc quyền dùng model; một số model yêu cầu
  subscription.
- Gemini 401/403: API key sai hoặc project chưa có quyền.
- HTTP 429: đã chạm quota/rate limit; thử lại sau.
- Embedding provider/model khác storage: chạy lại `python ingest.py`.
- Dữ liệu FAQ đã thay đổi: chạy lại `python ingest.py`.
- Kết quả reject chưa phù hợp: hiệu chỉnh threshold bằng tập evaluation.

## Giới hạn và hướng phát triển

MVP chưa có authentication, Redis, vector database, hybrid search hoặc
reranker. Hướng nâng cấp gồm FastAPI, Redis, PostgreSQL, Qdrant/Elasticsearch,
rate limiting, cache, monitoring và ingestion tự động từ CMS.
