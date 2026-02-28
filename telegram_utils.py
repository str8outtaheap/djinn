from __future__ import annotations

import logging
from typing import Any

from telegram import InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application

from config import MAX_TELEGRAM_CHUNK

LOGGER = logging.getLogger(__name__)


async def _send_once(
    application: Application,
    *,
    chat_id: int,
    text: str,
    reply_to_message_id: int | None = None,
    disable_notification: bool = False,
    reply_markup: InlineKeyboardMarkup | None = None,
    markdown: bool = True,
) -> int | None:
    kwargs: dict[str, Any] = {
        "chat_id": chat_id,
        "text": text,
        "disable_notification": disable_notification,
    }
    if reply_to_message_id is not None:
        kwargs["reply_to_message_id"] = reply_to_message_id
    if reply_markup is not None:
        kwargs["reply_markup"] = reply_markup
    if markdown:
        kwargs["parse_mode"] = ParseMode.MARKDOWN

    sent = await application.bot.send_message(**kwargs)
    message_id = getattr(sent, "message_id", None)
    if isinstance(message_id, int):
        return message_id
    return None


async def send_message(
    application: Application,
    text: str,
    *,
    chat_id: int,
    reply_to_message_id: int | None = None,
    disable_notification: bool = False,
    reply_markup: InlineKeyboardMarkup | None = None,
    prefer_markdown: bool = True,
) -> int | None:
    chunks = [text[i : i + MAX_TELEGRAM_CHUNK] for i in range(0, len(text), MAX_TELEGRAM_CHUNK)]
    if not chunks:
        chunks = [""]

    first_message_id: int | None = None
    for index, chunk in enumerate(chunks):
        chunk_reply_to = reply_to_message_id if index == 0 else None
        chunk_markup = reply_markup if index == 0 else None
        try:
            message_id = await _send_once(
                application,
                chat_id=chat_id,
                text=chunk,
                reply_to_message_id=chunk_reply_to,
                disable_notification=disable_notification,
                reply_markup=chunk_markup,
                markdown=prefer_markdown,
            )
        except BadRequest:
            message_id = await _send_once(
                application,
                chat_id=chat_id,
                text=chunk,
                reply_to_message_id=chunk_reply_to,
                disable_notification=disable_notification,
                reply_markup=chunk_markup,
                markdown=False,
            )
        except Exception as exc:
            LOGGER.warning(
                "failed to send telegram message chunk %s/%s: %s",
                index + 1,
                len(chunks),
                exc,
            )
            continue

        if first_message_id is None:
            first_message_id = message_id

    return first_message_id


async def edit_message(
    application: Application,
    *,
    chat_id: int,
    message_id: int,
    text: str,
    reply_markup: InlineKeyboardMarkup | None = None,
    prefer_markdown: bool = True,
) -> None:
    body = text
    if len(body) > MAX_TELEGRAM_CHUNK:
        body = body[: MAX_TELEGRAM_CHUNK - 1] + "…"

    kwargs: dict[str, Any] = {
        "chat_id": chat_id,
        "message_id": message_id,
        "text": body,
    }
    if reply_markup is not None:
        kwargs["reply_markup"] = reply_markup

    try:
        if prefer_markdown:
            await application.bot.edit_message_text(parse_mode=ParseMode.MARKDOWN, **kwargs)
            return
        await application.bot.edit_message_text(**kwargs)
    except BadRequest:
        try:
            await application.bot.edit_message_text(**kwargs)
        except BadRequest:
            LOGGER.debug("telegram message %s could not be edited", message_id)


async def delete_message(application: Application, *, chat_id: int, message_id: int) -> None:
    try:
        await application.bot.delete_message(chat_id=chat_id, message_id=message_id)
    except BadRequest:
        LOGGER.debug("telegram message %s could not be deleted", message_id)
    except Exception as exc:
        LOGGER.warning("failed to delete telegram message %s: %s", message_id, exc)
