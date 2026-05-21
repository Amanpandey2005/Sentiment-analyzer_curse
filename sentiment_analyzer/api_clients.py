"""Optional OpenAI-compatible API client for sentiment labeling/summaries."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


class OpenAICompatibleSentimentClient:
    """Minimal chat completions client using only the standard library."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is required for API sentiment analysis.")

    def analyze(self, review_text: str) -> dict[str, str]:
        prompt = (
            "Classify this e-commerce review as positive, neutral, or negative. "
            "Also provide a one-sentence summary. Return strict JSON with keys "
            "sentiment and summary.\n\n"
            f"Review: {review_text}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You analyze customer feedback for e-commerce teams."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Sentiment API request failed: {exc}") from exc

        content = body["choices"][0]["message"]["content"]
        return json.loads(content)
