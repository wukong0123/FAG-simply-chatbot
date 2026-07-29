"""Chainlit user interface for the FAQ chatbot."""

import logging

import chainlit as cl

from config import settings
from embedding_client import EmbeddingServiceError, create_embedding_client
from ollama_client import OllamaClient, OllamaServiceError
from retriever import Retriever, StorageError
from utils import is_contextual_follow_up, normalize_retrieval_query

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
LOGGER = logging.getLogger(__name__)
FALLBACK = "Tôi chưa tìm thấy thông tin phù hợp trong dữ liệu hiện có."
MAX_HISTORY_MESSAGES = 8
DIRECT_ANSWER_THRESHOLD = 0.90


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialize dependencies and greet the user."""
    try:
        client = OllamaClient()
        client.check_server()
        retriever = Retriever(client=create_embedding_client())
        cl.user_session.set("client", client)
        cl.user_session.set("retriever", retriever)
        cl.user_session.set("history", [])
        await cl.Message(
            content=(
                "Xin chào! Tôi có thể trả lời các câu hỏi FAQ nội bộ. "
                f"Model hiện tại: `{settings.chat_model}`."
            )
        ).send()
    except (
        OllamaServiceError,
        EmbeddingServiceError,
        StorageError,
        ValueError,
        OSError,
    ) as exc:
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
    history: list[dict[str, str]] = cl.user_session.get("history") or []
    if retriever is None or client is None:
        await cl.Message(content="Chatbot chưa sẵn sàng. Hãy tải lại trang.").send()
        return
    response = cl.Message(content="Đang tìm thông tin phù hợp...")
    await response.send()
    try:
        retrieval_question = question
        if is_contextual_follow_up(question, bool(history)):
            try:
                retrieval_question = client.rewrite_question(question, history)
                LOGGER.info("Đã viết lại câu hỏi nối tiếp để retrieval.")
            except OllamaServiceError as exc:
                LOGGER.warning("Không thể viết lại câu hỏi nối tiếp: %s", exc)
        retrieval_question = normalize_retrieval_query(retrieval_question)
        matches, rejected = retriever.retrieve(retrieval_question)
        if rejected:
            response.content = FALLBACK
            answer_for_history = FALLBACK
        else:
            top_score = matches[0]["score"]
            relevant_matches = [
                item
                for item in matches
                if item["score"] >= max(settings.similarity_threshold, top_score - 0.15)
            ]
            if top_score >= DIRECT_ANSWER_THRESHOLD:
                answer = matches[0]["answer"]
            else:
                try:
                    answer = client.chat(
                        question, relevant_matches, history=history
                    )
                except OllamaServiceError as exc:
                    LOGGER.warning("LLM lỗi, dùng câu trả lời FAQ gần nhất: %s", exc)
                    answer = matches[0]["answer"]
            sources = "\n".join(
                f"- {item['question']} — độ phù hợp {item['score']:.2f}"
                for item in relevant_matches
            )
            response.content = f"{answer}\n\nNguồn tham khảo:\n{sources}"
            answer_for_history = answer
    except (ValueError, StorageError, OllamaServiceError, EmbeddingServiceError) as exc:
        LOGGER.exception("Xử lý câu hỏi thất bại")
        response.content = str(exc)
        answer_for_history = response.content
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_for_history},
        ]
    )
    cl.user_session.set("history", history[-MAX_HISTORY_MESSAGES:])
    await response.update()
