"""Deepseek API client built on the OpenAI-compatible SDK."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_URL = "https://api.deepseek.com"


class DeepseekClient:
    """A thin OpenAI-compatible client wrapper for Deepseek chat calls."""

    def __init__(self) -> None:
        """Load credentials from ``.env`` and initialize the OpenAI SDK."""

        env_path = PROJECT_ROOT / ".env"
        load_dotenv(dotenv_path=env_path, override=False)

        api_key = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
        base_url = (os.getenv("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL).strip()
        base_url = base_url.strip("`").strip().rstrip("/")

        if not api_key or api_key == "your_deepseek_api_key_here":
            raise ValueError(
                "DEEPSEEK_API_KEY is missing. Please create `.env` from `.env.example` "
                "and fill in a valid Deepseek API key."
            )

        if not base_url:
            base_url = DEFAULT_BASE_URL

        self.api_key = api_key
        self.base_url = base_url
        self.model_name = "deepseek-chat"
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=30.0,
            max_retries=2,
        )

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Call the Deepseek chat API and return the generated content."""

        if not prompt or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string.")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt.strip(),
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("Deepseek returned an empty response.")
            return content.strip()
        except APITimeoutError as exc:
            raise RuntimeError(
                "Deepseek API request timed out. Please retry later."
            ) from exc
        except RateLimitError as exc:
            raise RuntimeError(
                "Deepseek API rate limit was reached. Please slow down and retry."
            ) from exc
        except APIConnectionError as exc:
            raise RuntimeError(
                "Failed to connect to the Deepseek API. Please check your network."
            ) from exc
        except APIStatusError as exc:
            raise RuntimeError(
                f"Deepseek API returned an error status: {exc.status_code}."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Deepseek API call failed unexpectedly: {exc}"
            ) from exc
