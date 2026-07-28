"""Chainlit user interface for the FAQ chatbot."""

import logging

import chainlit as cl

from config import settings
from ollama_client import OllamaClient, OllamaServiceError
from retriever import Retriever, StorageError

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)
FALLBACK = "Tôi chưa tìm thấy thông tin phù hợp trong dữ liệu hiện có."


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize dependencies and greet the user."""
    try:
        client = OllamaClient()
        client.check_server()
        retriever = Retriever(client=client)
        cl.user_session.set("client", client)
        cl.user_session.set("retriever", retriever)
        await cl.Message(
            content=(
                "Xin chào! Tôi có thể trả lời các câu hỏi FAQ nội bộ. "
                f"Model hiện tại: `{settings.chat_model}`."
            )
        ).send()
    except (OllamaServiceError, StorageError, ValueError, OSError) as exc:
        LOGGER.exception("Không thể khởi tạo ứng dụng")
        await cl.Message(content=f"Không thể khởi tạo chatbot:\n{exc}").send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Retrieve FAQ context and generate a grounded response."""
    question = message.content.strip()
    if not question:
        await cl.Message(content="Vui lòng nhập câu hỏi.").send()
        return
    retriever: Retriever | None = cl.user_session.get("retriever")
    client: OllamaClient | None = cl.user_session.get("client")
    if retriever is None or client is None:
        await cl.Message(content="Chatbot chưa sẵn sàng. Hãy tải lại trang.").send()
        return
    response = cl.Message(content="Đang tìm thông tin phù hợp...")
    await response.send()
    try:
        matches, rejected = retriever.retrieve(question)
        if rejected:
            response.content = FALLBACK
        else:
            top_score = matches[0]["score"]
            relevant_matches = [
                item
                for item in matches
                if item["score"] >= max(settings.similarity_threshold, top_score - 0.15)
            ]
            if top_score >= 0.80:
                answer = matches[0]["answer"]
            else:
                try:
                    answer = client.chat(question, relevant_matches)
                except OllamaServiceError as exc:
                    LOGGER.warning("LLM lỗi, dùng câu trả lời FAQ gần nhất: %s", exc)
                    answer = matches[0]["answer"]
            sources = "\n".join(
                f"- {item['question']} — độ phù hợp {item['score']:.2f}"
                for item in relevant_matches
            )
            response.content = f"{answer}\n\nNguồn tham khảo:\n{sources}"
    except (ValueError, StorageError, OllamaServiceError) as exc:
        LOGGER.exception("Xử lý câu hỏi thất bại")
        response.content = str(exc)
    await response.update()
