# FAQ Chatbot

Chatbot trả lời các câu hỏi FAQ bằng tìm kiếm ngữ nghĩa. Dự án sử dụng
Gemini để tạo embedding, Ollama Cloud để tạo câu trả lời và Chainlit cho giao
diện chat.

## Chuẩn bị

- Cài đặt Anaconda
- Tạo môi trường Python 3.11 trong Anaconda và kích hoạt môi trường đó.
- Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

- Cập nhật file `.env` với `OLLAMA_API_KEY` và `GEMINI_API_KEY` trước khi chạy
  ứng dụng. (hiện tại đang chạy trên phần cứng nên không cần key Ollama hay Gemini)

## Chạy ứng dụng

Tạo dữ liệu embedding từ file FAQ:

```bash
python ingest.py
```

Khởi động giao diện chatbot:

```bash
python -m chainlit run app.py
```

Sau khi khởi động, mở địa chỉ được Chainlit hiển thị trên trình duyệt.

## Dữ liệu FAQ

Danh sách câu hỏi và câu trả lời nằm trong file `data/faqs.json`. Sau khi thay
đổi file này, hãy chạy lại `python ingest.py` để cập nhật embedding.
