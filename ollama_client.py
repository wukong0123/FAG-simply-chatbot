"""Small Ollama API wrapper with user-friendly errors."""

import logging
from typing import Any

from config import Settings, settings

LOGGER = logging.getLogger(__name__)


class OllamaServiceError(RuntimeError):
    """An actionable Ollama service error."""


class OllamaClient:
    """Call Ollama for health checks, embeddings, and chat."""

    def __init__(self, config: Settings = settings) -> None:
        try:
            import ollama
        except ImportError as exc:
            raise OllamaServiceError(
                "Chưa cài package ollama. Hãy chạy: pip install -r requirements.txt"
            ) from exc
        self.config = config
        if config.ollama_base_url.startswith("https://ollama.com") and not config.ollama_api_key:
            raise OllamaServiceError(
                "Thiếu OLLAMA_API_KEY để gọi Ollama Cloud. Hãy cấu hình trong file .env."
            )
        headers = (
            {"Authorization": f"Bearer {config.ollama_api_key}"}
            if config.ollama_api_key
            else None
        )
        self.client = ollama.Client(
            host=config.ollama_base_url,
            timeout=config.request_timeout_seconds,
            headers=headers,
        )

    def _friendly_error(self, exc: Exception, model: str | None = None) -> OllamaServiceError:
        status = getattr(exc, "status_code", None)
        text = str(exc).lower()
        if status == 403 or "403" in text:
            return OllamaServiceError(
                "Model cloud yêu cầu quyền truy cập hoặc subscription (HTTP 403)."
            )
        if status == 429 or "429" in text:
            return OllamaServiceError(
                "Đã vượt quota hoặc rate limit (HTTP 429). Vui lòng thử lại sau."
            )
        if "not found" in text and model:
            return OllamaServiceError(
                f"Không tìm thấy model {model}.\nHãy chạy: ollama pull {model}"
            )
        if "connect" in text or "refused" in text:
            return OllamaServiceError(
                f"Không thể kết nối tới Ollama tại {self.config.ollama_base_url}.\n"
                "Hãy mở Ollama hoặc chạy: ollama serve"
            )
        return OllamaServiceError(f"Ollama trả về lỗi: {exc}")

    def check_server(self) -> None:
        """Check that Ollama is reachable."""
        try:
            self.client.list()
        except Exception as exc:
            raise self._friendly_error(exc) from exc

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """Create embeddings and return them as a batch."""
        try:
            response = self.client.embed(
                model=self.config.embedding_model, input=texts
            )
            vectors = response.get("embeddings", [])
            if not vectors:
                raise OllamaServiceError("Ollama trả về embedding rỗng.")
            return vectors
        except OllamaServiceError:
            raise
        except Exception as exc:
            raise self._friendly_error(exc, self.config.embedding_model) from exc

    def chat(
        self,
        question: str,
        matches: list[dict[str, Any]],
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate a grounded Vietnamese answer."""
        context_blocks = []
        for index, item in enumerate(matches, start=1):
            context_blocks.append(
                f"[FAQ {index}]\nCâu hỏi: {item['question']}\n"
                f"Câu trả lời: {item['answer']}"
            )
        system = (
            "Bạn là chatbot FAQ nội bộ.\n\n"
            "Chỉ trả lời dựa trên phần CONTEXT được cung cấp. "
            "Không tự thêm chính sách, số liệu, tên người hoặc thông tin không có "
            "trong CONTEXT. Nếu CONTEXT không đủ để trả lời, hãy nói: "
            '"Tôi chưa tìm thấy thông tin phù hợp trong dữ liệu hiện có."\n\n'
            "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng. Nếu có nhiều thông tin "
            "liên quan, trình bày theo từng ý. Không nhắc đến embedding, vector "
            "hoặc thuật toán nội bộ cho người dùng cuối."
        )
        user = (
            f"CONTEXT:\n{chr(10).join(context_blocks)}\n\n"
            f"CÂU HỎI CỦA NGƯỜI DÙNG:\n{question}"
        )
        recent_history = (history or [])[-8:]
        messages = [{"role": "system", "content": system}]
        messages.extend(
            {"role": item["role"], "content": item["content"]}
            for item in recent_history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        )
        messages.append({"role": "user", "content": user})
        try:
            response = self.client.chat(
                model=self.config.chat_model,
                messages=messages,
                options={"temperature": 0.1},
            )
            content = response["message"]["content"].strip()
            if not content:
                raise OllamaServiceError("Chat model trả về nội dung rỗng.")
            return content
        except OllamaServiceError:
            raise
        except Exception as exc:
            raise self._friendly_error(exc, self.config.chat_model) from exc

    def rewrite_question(
        self, question: str, history: list[dict[str, str]]
    ) -> str:
        """Rewrite a contextual follow-up as a standalone retrieval query."""
        recent_history = history[-6:]
        conversation = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in recent_history
            if item.get("role") in {"user", "assistant"} and item.get("content")
        )
        prompt = (
            "Dựa vào lịch sử, hãy viết lại CÂU HỎI MỚI thành một câu hỏi độc lập "
            "để tìm kiếm tài liệu. Thay đại từ và từ viết tắt bằng tên đối tượng "
            "đầy đủ đã được xác định trong lịch sử. Nếu câu hỏi có nhiều ý, phải "
            "giữ lại đầy đủ mọi ý. Trong chatbot này, WC hoặc World Cup không nêu "
            "năm được hiểu là FIFA World Cup 2026. Giữ nguyên ngôn ngữ người dùng. "
            "Chỉ xuất câu hỏi đã viết lại, không trả lời và không giải thích.\n\n"
            f"LỊCH SỬ:\n{conversation}\n\nCÂU HỎI MỚI:\n{question}"
        )
        try:
            response = self.client.chat(
                model=self.config.chat_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            rewritten = response["message"]["content"].strip().strip('"')
            return rewritten or question
        except Exception as exc:
            raise self._friendly_error(exc, self.config.chat_model) from exc
