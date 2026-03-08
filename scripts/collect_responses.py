#!/usr/bin/env python3
"""
collect_responses.py

Production-ish response collection script for a prompt dataset.

Key features
- Accepts prompts JSON in either format:
  (A) list[dict]  -> [{"id":..., "prompt":..., "category":...}, ...]
  (B) dict[str, list[dict]] -> {"math_basic": [...], "commonsense": [...]}
- Robust error handling + exponential backoff retries
- Conservative rate limiting via --min-interval (seconds between requests)
- Checkpointing to CSV every N prompts
- Saves metadata logs:
  output_dir/metadata/query_patterns.csv
  output_dir/metadata/errors.csv
- Reproducible selection (seed), optional shuffle, optional balanced sampling

Usage examples
1) Small test run:
   python scripts/collect_responses.py --prompts data/prompts_v1.json --output data/responses_test.csv --limit 10 --shuffle

2) Balanced 400 (80 per category if you have 5 categories):
   python scripts/collect_responses.py --prompts data/prompts_v1.json --output data/responses_openai.csv --limit 400 --balanced --shuffle

Prereqs
- openai>=1.x, python-dotenv, pandas installed
- .env contains OPENAI_API_KEY=...
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

try:
    # These exist in openai>=1.x
    from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
except Exception:  # pragma: no cover
    APIConnectionError = APIError = APITimeoutError = RateLimitError = Exception


# -------------------------
# Utilities
# -------------------------

def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("collector")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers if re-run in same interpreter
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_dir / "collection.log", encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_prompts(path: Path) -> List[Dict[str, Any]]:
    """
    Load prompts from JSON.

    Supports:
      - list[dict]  (recommended)
      - dict[str, list[dict]] (legacy: category -> list of prompts)
    Normalizes to list[dict] and ensures minimally required fields exist.
    """
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        prompts = data
    elif isinstance(data, dict):
        prompts = []
        for v in data.values():
            if not isinstance(v, list):
                raise TypeError("Invalid prompts JSON: dict values must be lists of prompt objects")
            prompts.extend(v)
    else:
        raise TypeError("Invalid prompts JSON: root must be a list or a dict")

    normalized: List[Dict[str, Any]] = []
    for i, p in enumerate(prompts):
        if not isinstance(p, dict):
            raise TypeError(f"Prompt #{i} is not a JSON object/dict")

        if "prompt" not in p:
            raise ValueError(f"Prompt #{i} missing required field 'prompt'")

        # Fill common metadata if absent
        if "category" not in p:
            p = {**p, "category": "unknown"}
        if "id" not in p:
            p = {**p, "id": f"prompt_{i:06d}"}

        normalized.append(p)

    return normalized


def basic_schema_check(prompts: List[Dict[str, Any]], logger: logging.Logger, sample_n: int = 20) -> None:
    required = {"id", "prompt", "category"}
    for i, p in enumerate(prompts[:sample_n]):
        missing = required - set(p.keys())
        if missing:
            raise ValueError(f"Prompt sample #{i} is missing fields {missing}. Keys={list(p.keys())}")

    logger.info("Schema check: PASS (sampled %d prompts)", min(sample_n, len(prompts)))


def select_prompts(
    prompts: List[Dict[str, Any]],
    limit: Optional[int],
    shuffle: bool,
    seed: int,
    balanced: bool,
    per_category: Optional[int],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    prompts_copy = list(prompts)

    if balanced:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for p in prompts_copy:
            groups.setdefault(str(p.get("category", "unknown")), []).append(p)

        cats = sorted(groups.keys())
        if not cats:
            return []

        if per_category is None:
            if limit is None:
                raise ValueError("Balanced sampling requires --limit or --per-category.")
            per_category = max(1, (limit + len(cats) - 1) // len(cats))  # ceil

        selected: List[Dict[str, Any]] = []
        for c in cats:
            g = groups[c]
            if shuffle:
                rng.shuffle(g)
            selected.extend(g[: min(per_category, len(g))])

        if limit is not None:
            selected = selected[:limit]

        logger.info(
            "Balanced sampling: categories=%d per_category=%d selected=%d",
            len(cats),
            per_category,
            len(selected),
        )
        return selected

    if shuffle:
        rng.shuffle(prompts_copy)

    if limit is not None:
        prompts_copy = prompts_copy[:limit]

    logger.info("Selected %d prompts (balanced=%s shuffle=%s)", len(prompts_copy), balanced, shuffle)
    return prompts_copy


# -------------------------
# Collector
# -------------------------

@dataclass
class CollectorConfig:
    model: str
    temperature: float
    max_tokens: int
    min_interval_s: float
    max_retries: int
    timeout_s: Optional[float]


class ResponseCollector:
    def __init__(self, client: OpenAI, cfg: CollectorConfig, logger: logging.Logger):
        self.client = client
        self.cfg = cfg
        self.logger = logger

        self._last_request_t = 0.0

        self.results: List[Dict[str, Any]] = []
        self.query_log: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []

    def _sleep_to_respect_interval(self) -> None:
        if self.cfg.min_interval_s <= 0:
            return
        now = time.time()
        elapsed = now - self._last_request_t
        if elapsed < self.cfg.min_interval_s:
            time.sleep(self.cfg.min_interval_s - elapsed)

    def _backoff(self, attempt: int) -> float:
        # Exponential backoff with jitter, capped at 60s
        base = min(60.0, (2 ** (attempt - 1)) * 2.0)  # 2,4,8,...
        jitter = random.random() * 0.25 * base
        return base + jitter

    def query_one(self, prompt_obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt_text = str(prompt_obj["prompt"])
        meta = {k: v for k, v in prompt_obj.items() if k != "prompt"}

        for attempt in range(1, self.cfg.max_retries + 1):
            self._sleep_to_respect_interval()
            start = time.time()
            ts = utc_now_iso()

            try:
                resp = self.client.chat.completions.create(
                    model=self.cfg.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful assistant. Provide a clear final answer. "
                                "If you use steps, keep them concise."
                            ),
                        },
                        {"role": "user", "content": prompt_text},
                    ],
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=self.cfg.timeout_s,
                )

                latency_ms = int((time.time() - start) * 1000)
                self._last_request_t = time.time()

                choice = resp.choices[0]
                content = choice.message.content if choice.message else ""
                usage = getattr(resp, "usage", None)

                out = {
                    "id": meta.get("id"),
                    "category": meta.get("category"),
                    "difficulty": meta.get("difficulty"),
                    "source": meta.get("source"),
                    "prompt": prompt_text,
                    "response": content,
                    "model": self.cfg.model,
                    "temperature": self.cfg.temperature,
                    "max_tokens": self.cfg.max_tokens,
                    "timestamp": ts,
                    "latency_ms": latency_ms,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "tokens_prompt": getattr(usage, "prompt_tokens", None),
                    "tokens_completion": getattr(usage, "completion_tokens", None),
                    "tokens_total": getattr(usage, "total_tokens", None),
                    "attempt": attempt,
                }

                self.query_log.append(
                    {
                        "timestamp": ts,
                        "id": meta.get("id"),
                        "category": meta.get("category"),
                        "success": True,
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                    }
                )
                return out

            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                latency_ms = int((time.time() - start) * 1000)
                self._last_request_t = time.time()
                wait_s = self._backoff(attempt)

                self.logger.warning(
                    "Transient error (%s) id=%s attempt=%d; backoff=%.1fs",
                    type(e).__name__,
                    meta.get("id"),
                    attempt,
                    wait_s,
                )
                self.error_log.append(
                    {
                        "timestamp": ts,
                        "id": meta.get("id"),
                        "category": meta.get("category"),
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                    }
                )

                if attempt < self.cfg.max_retries:
                    time.sleep(wait_s)
                    continue
                return None

            except APIError as e:
                # Treat as retryable up to max_retries
                latency_ms = int((time.time() - start) * 1000)
                self._last_request_t = time.time()
                wait_s = self._backoff(attempt)

                self.logger.warning(
                    "API error (%s) id=%s attempt=%d; backoff=%.1fs",
                    type(e).__name__,
                    meta.get("id"),
                    attempt,
                    wait_s,
                )
                self.error_log.append(
                    {
                        "timestamp": ts,
                        "id": meta.get("id"),
                        "category": meta.get("category"),
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                    }
                )

                if attempt < self.cfg.max_retries:
                    time.sleep(wait_s)
                    continue
                return None

            except Exception as e:
                latency_ms = int((time.time() - start) * 1000)
                self._last_request_t = time.time()
                self.logger.error(
                    "Non-retryable error (%s) id=%s: %s",
                    type(e).__name__,
                    meta.get("id"),
                    str(e),
                )
                self.error_log.append(
                    {
                        "timestamp": ts,
                        "id": meta.get("id"),
                        "category": meta.get("category"),
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "attempt": attempt,
                        "latency_ms": latency_ms,
                    }
                )
                return None

    def save_checkpoint(self, output_csv: Path) -> None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.results).to_csv(output_csv, index=False)


# -------------------------
# Main
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Collect LLM responses for a prompt dataset.")
    parser.add_argument("--prompts", required=True, help="Path to prompts JSON")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None, help="Max number of prompts to query")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before selecting prompts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--balanced", action="store_true", help="Sample roughly evenly per category")
    parser.add_argument("--per-category", type=int, default=None, help="If balanced, how many per category")

    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name (edit as needed)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--min-interval", type=float, default=20.0, help="Minimum seconds between requests")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout seconds")

    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save every N processed prompts")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Env var name for API key")
    parser.add_argument("--dotenv", default=".env", help="Path to .env file")

    args = parser.parse_args()

    load_dotenv(args.dotenv)
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key: set {args.api_key_env} in {args.dotenv} or environment.")

    output_csv = Path(args.output)
    logger = setup_logging(output_csv.parent / "logs", args.log_level)

    prompts_path = Path(args.prompts)
    prompts = load_prompts(prompts_path)
    logger.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    basic_schema_check(prompts, logger)

    selected = select_prompts(
        prompts,
        limit=args.limit,
        shuffle=args.shuffle,
        seed=args.seed,
        balanced=args.balanced,
        per_category=args.per_category,
        logger=logger,
    )
    if not selected:
        logger.error("No prompts selected; exiting.")
        sys.exit(1)

    cfg = CollectorConfig(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        min_interval_s=args.min_interval,
        max_retries=args.max_retries,
        timeout_s=args.timeout,
    )

    client = OpenAI(api_key=api_key)
    collector = ResponseCollector(client, cfg, logger)

    start_ts = datetime.utcnow()
    logger.info(
        "Starting collection: n=%d model=%s min_interval=%.1fs retries=%d",
        len(selected),
        cfg.model,
        cfg.min_interval_s,
        cfg.max_retries,
    )

    for i, p in enumerate(selected, start=1):
        res = collector.query_one(p)
        if res is not None:
            collector.results.append(res)

        if i % args.checkpoint_every == 0:
            collector.save_checkpoint(output_csv)
            logger.info(
                "Checkpoint saved: processed=%d/%d successes=%d",
                i,
                len(selected),
                len(collector.results),
            )

    # Final save
    collector.save_checkpoint(output_csv)

    meta_dir = output_csv.parent / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(collector.query_log).to_csv(meta_dir / "query_patterns.csv", index=False)
    pd.DataFrame(collector.error_log).to_csv(meta_dir / "errors.csv", index=False)

    elapsed = datetime.utcnow() - start_ts
    logger.info(
        "Done. successes=%d/%d elapsed=%s output=%s",
        len(collector.results),
        len(selected),
        elapsed,
        output_csv,
    )
    logger.info("Metadata written to: %s", meta_dir)


if __name__ == "__main__":
    main()
