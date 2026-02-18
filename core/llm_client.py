import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

from core.key_manager import get_key_manager

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6").strip()
MAX_RETRIES = 4
BASE_DELAY = 1.2
TIMEOUT = 60


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    model: str
    key_alias: str
    latency_ms: float


def chat(
    system: str,
    messages: List[dict],
    max_tokens: int = 512,
    temperature: float = 0.85,
    thinking: bool = False,
    thinking_budget: int = 800,
) -> str:
    return chat_full(
        system=system,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking=thinking,
        thinking_budget=thinking_budget,
    ).text


def chat_full(
    system: str,
    messages: List[dict],
    max_tokens: int = 512,
    temperature: float = 0.85,
    thinking: bool = False,
    thinking_budget: int = 800,
) -> LLMResponse:
    km = get_key_manager()
    last_error: Optional[Exception] = None

    for attempt in range(MAX_RETRIES):
        key_record = km.acquire()

        payload_dict: dict = {
            "model": MODEL,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }

        if thinking:
            payload_dict["thinking"] = {
                "type": "enabled",
                "budget_tokens": max(1, min(thinking_budget, max_tokens - 100)),
            }
        else:
            payload_dict["temperature"] = temperature

        payload_bytes = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8")

        headers = {
            "x-api-key": key_record.key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json; charset=utf-8",
        }

        t0 = time.monotonic()
        try:
            req = urllib.request.Request(
                ANTHROPIC_URL,
                data=payload_bytes,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                raw = resp.read()

            latency_ms = (time.monotonic() - t0) * 1000.0
            body = json.loads(raw.decode("utf-8"))

            usage = body.get("usage", {})
            tokens_in = int(usage.get("input_tokens", 0))
            tokens_out = int(usage.get("output_tokens", 0))

            km.report_usage(key_record.alias, tokens_in, tokens_out)

            text_parts = [
                block["text"]
                for block in body.get("content", [])
                if block.get("type") == "text"
            ]
            text = "\n".join(text_parts).strip()

            return LLMResponse(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                model=body.get("model", MODEL),
                key_alias=key_record.alias,
                latency_ms=latency_ms,
            )

        except urllib.error.HTTPError as exc:
            status = exc.code
            km.report_error(key_record.alias, status)

            if status in (429, 529) or status >= 500:
                jitter = random.uniform(0.0, 0.8)
                wait = BASE_DELAY * (2 ** attempt) + jitter
                time.sleep(wait)
                last_error = exc
                continue

            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = str(exc)
            raise RuntimeError("API error {}: {}".format(status, err_body)) from exc

        except urllib.error.URLError as exc:
            km.report_error(key_record.alias, 503)
            time.sleep(BASE_DELAY * (2 ** attempt))
            last_error = exc
            continue

        except (KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("Unexpected API response format: {}".format(exc)) from exc

    raise RuntimeError(
        "API call failed after {} attempts. Last error: {}".format(MAX_RETRIES, last_error)
    )
