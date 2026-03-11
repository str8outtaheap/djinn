from __future__ import annotations

import logging
import os

from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

import commands
import config
import state
import telegram_utils

# Re-export for compatibility with existing tests/importers.
BotState = state.BotState
ProgressState = state.ProgressState
QueuedTurn = state.QueuedTurn

load_runtime_state = state.load_runtime_state
persist_state = state.persist_state

send_message = telegram_utils.send_message

resolve_workdir = commands.resolve_workdir
build_prompt = commands.build_prompt
build_help_text = commands.build_help_text
is_authorized = commands.is_authorized
format_progress_line = commands.format_progress_line
note_progress_event = commands.note_progress_event
render_progress = commands.render_progress
format_changed_files_summary = commands.format_changed_files_summary
truncate_output = commands.truncate_output
latest_sessions_by_workdir = commands.latest_sessions_by_workdir
format_sessions_text = commands.format_sessions_text
resolve_session_selection = commands.resolve_session_selection
drain_queued_turns = commands.drain_queued_turns
run_turn_for_input = commands.run_turn_for_input
sessions_cmd = commands.sessions_cmd
sessions_callback = commands.sessions_callback
on_message = commands.on_message
startup_notify = commands.startup_notify
get_codex_client = commands.get_codex_client

TELEGRAM_CHAT_ID = config.TELEGRAM_CHAT_ID
TELEGRAM_USER_ID = config.TELEGRAM_USER_ID
STATE_PATH = config.STATE_PATH
PROGRESS_MAX_ACTIONS = config.PROGRESS_MAX_ACTIONS
SESSIONS_FETCH_LIMIT = config.SESSIONS_FETCH_LIMIT
SESSIONS_LIST_LIMIT = config.SESSIONS_LIST_LIMIT

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)
# Avoid leaking bot token in HTTP URL logs from lower-level clients.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# Entrypoint

def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        raise SystemExit("Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID in .env")

    bot_state = state.BotState()
    runtime_state = state.load_runtime_state()

    saved_workdir = runtime_state.get("workdir")
    if saved_workdir and os.path.isdir(saved_workdir):
        bot_state.workdir = saved_workdir

    saved_thread = runtime_state.get("thread_id")
    if saved_thread:
        bot_state.thread_id = saved_thread

    application = (
        ApplicationBuilder()
        .token(config.TELEGRAM_BOT_TOKEN)
        .concurrent_updates(8)
        .post_init(commands.startup_notify)
        .post_shutdown(commands.shutdown)
        .build()
    )
    application.bot_data["state"] = bot_state

    application.add_handler(CommandHandler("start", commands.start_cmd))
    application.add_handler(CommandHandler("help", commands.help_cmd))
    application.add_handler(CommandHandler("cancel", commands.cancel_cmd))
    application.add_handler(CommandHandler("reset", commands.reset_cmd))
    application.add_handler(CommandHandler("cd", commands.cd_cmd))
    application.add_handler(CommandHandler("status", commands.status_cmd))
    application.add_handler(CommandHandler("sessions", commands.sessions_cmd))
    application.add_handler(CommandHandler("run", commands.run_cmd))
    application.add_handler(CallbackQueryHandler(commands.sessions_callback, pattern=r"^sessions:"))
    application.add_handler(
        MessageHandler((filters.TEXT | filters.VOICE) & ~filters.COMMAND, commands.on_message)
    )

    application.run_polling()


if __name__ == "__main__":
    main()
