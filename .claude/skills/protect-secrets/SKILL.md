---
name: protect-secrets
description: Prevents Claude from reading .env files or leaking API keys, secrets, passwords, or any credentials present in the codebase or tool outputs.
---

# Protect Secrets

## Rules (always active, no exceptions)

1. **Never read `.env` files.** If a task requires knowing the contents of `.env`, ask the user to share only the specific variable name — never the value.

2. **Never print, log, or include API keys, secrets, tokens, or passwords in any response.** This includes values that appear in tool outputs, file contents, or command results.

3. **If a secret appears in a tool result**, redact it immediately in your response. Replace the value with `[REDACTED]`. Example:
   - Raw output: `ALPACA_API_KEY=abc123xyz`
   - Response: `ALPACA_API_KEY=[REDACTED]`

4. **Never suggest hardcoding secrets** in source files. Always direct secrets to `.env` and read them via `os.getenv()` or `python-dotenv`.

5. **Never commit `.env` files.** If asked to run `git add` on files containing secrets, refuse and remind the user that `.env` must be in `.gitignore`.

6. **If `.env` is not in `.gitignore`**, warn the user immediately and add it before doing anything else.

## Files considered secret (never read or display contents)

- `.env`, `.env.local`, `.env.production`, `.env.*`
- `*.pem`, `*.key`, `*.p12`, `*.pfx`
- `secrets.json`, `credentials.json`, `service-account.json`
- Any file with "secret", "credential", or "token" in the name

## Safe pattern to always follow

```python
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("MY_API_KEY")  # never hardcode the value
```