# djinn

Telegram bot bridge for the [Codex CLI](https://developers.openai.com/codex/cli), powered by `codex app-server` (JSON-RPC).

## Features
- Single authorized chat controls a stateful Djinn thread
- Uses Codex App Server threads (`thread/start`, `thread/resume`, `turn/start`)
- Live progress updates from structured item lifecycle events (`started`, `delta`, `completed`)
- Telegram approval buttons for command/file-change approvals
- Startup health ping ("Djinn online") to the authorized chat
- Session browser and quick resume (`/sessions`, `/sessions here`, `/sessions use`)
- `/cd`, `/status`, `/reset`, `/pin`, `/unpin`, `/run`, `/proj`
- Optional voice note transcription via OpenAI SDK

## Quickstart
Prereqs: Codex CLI on `PATH`.

```sh
uv sync --all-groups
cp .env.example .env
uv run python bot.py
```

## Configuration
Required:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID` (single chat allowed)

Optional:
- `TELEGRAM_USER_ID` (strongly recommended in group chats)
- `CODEX_CMD` (default: `codex`)
  On Apple Silicon, if subprocesses run under Rosetta, set `CODEX_CMD=arch -arm64 codex`.
- `CODEX_APPROVAL_POLICY` (default: `on-request`)
- `CODEX_WORKDIR` (default: launch directory)
- `VOICE_TRANSCRIPTION` (`true` to enable)
- `OPENAI_API_KEY` (required for voice transcription)
- `OPENAI_TRANSCRIPTION_MODEL` (default: `gpt-4o-mini-transcribe`)

Codex CLI config (optional): `~/.codex/config.toml`
```toml
sandbox_mode = "workspace-write"
approval_policy = "on-request"

[sandbox_workspace_write]
network_access = true
```

## Commands
- `/start`: confirm connectivity
- `/help`: show command help
- `/cancel`: cancel the active turn or `/run` command
- `/last`: resend the latest turn result
- `/approve`: approve the most recent pending action
- `/deny`: deny the most recent pending action
- `/cd <path>`: change working directory (supports relative paths)
- `/status`: show thread state and working directory
- `/proj`: list project aliases
- `/proj <name>`: switch to a saved project context (workdir + thread + pin)
- `/proj <name> <path>`: save (and switch to) a project context
- `/proj rm <name>`: remove a project alias
- `/sessions`: list the newest session per workdir
- `/sessions here`: list recent sessions for the current workdir
- `/sessions use <n|thread_id>`: switch to a listed session or explicit thread id
- `/pin <text>`: set a short context note shown during runs
- `/pin`: show current pin
- `/unpin`: clear the pin
- `/run <command>`: run a command in the current working directory (time-limited and output-capped for mobile usability)

When Djinn is busy, the latest incoming message is queued (one deep) and runs next.

Advanced/debug command:
- `/reset`: clear current Djinn thread

## Development
Run checks with `uv run`:

```sh
uv run pytest
uv run ruff check .
uv run ty check .
```

## State Files
- Runtime state: `~/.djinn/state.json`
- Project contexts: `~/.djinn/projects.json` (`path`, `thread_id`, `pin`)

## Security Notes
This bot can run shell commands and edit files.

- Keep the bot token private.
- Restrict `TELEGRAM_CHAT_ID` to a trusted chat.
- In group chats, always set `TELEGRAM_USER_ID`; otherwise any chat member can issue commands and approve actions.

## License
MIT
