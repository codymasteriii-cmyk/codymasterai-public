"""
TokenOps Safeguards — illustrative implementation
Two patterns: intelligent model routing + semantic caching.

Requires: pip install anthropic
API key:  set ANTHROPIC_API_KEY in your environment
"""

import anthropic
import hashlib
import time

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment


# ── 1. INTELLIGENT MODEL ROUTER ───────────────────────────────────────────────
#
# Most enterprise chatbot deployments send every query to a frontier model
# regardless of complexity. This router intercepts each request and selects
# the cheapest model capable of handling the task — invisible to the user,
# significant in aggregate cost impact.
#
# Routing alone can reduce token spend 40–70% on mixed-workload deployments.

MODELS = {
    "haiku":  "claude-haiku-4-5",   # $1  / $5  per 1M tokens — fast, cheap
    "sonnet": "claude-sonnet-4-6",  # $3  / $15 per 1M tokens — balanced
    "opus":   "claude-opus-4-8",    # $5  / $25 per 1M tokens — deep reasoning
}

# Banking task signals that indicate frontier-model reasoning is warranted
COMPLEX_SIGNALS = {
    "regulatory gap", "stress test", "capital requirement",
    "legal opinion", "model validation", "strategic review",
    "audit finding", "credit assessment", "risk appetite",
    "counterparty exposure", "liquidity ratio",
}

# Banking task signals that indicate a lightweight model is sufficient
SIMPLE_SIGNALS = {
    "extract", "summarize meeting", "action items", "format this",
    "classify", "find the date", "list all", "convert to",
    "meeting minutes", "reformat", "spell check",
}


def route_query(prompt: str, has_large_attachment: bool = False) -> str:
    """
    Return the model ID appropriate for this query.
    Priority order: explicit complex signal > attachment heuristic > simple
    signal > safe default (Sonnet).
    """
    prompt_lower = prompt.lower()

    # Complex regulatory or risk tasks warrant Opus regardless of length
    if any(signal in prompt_lower for signal in COMPLEX_SIGNALS):
        return MODELS["opus"]

    # Large attachments paired with a substantive prompt → Sonnet floor.
    # Haiku may miss nuance when the retrieved context is lengthy.
    if has_large_attachment and len(prompt) > 300:
        return MODELS["sonnet"]

    # Explicit simple/extraction tasks → Haiku
    if any(signal in prompt_lower for signal in SIMPLE_SIGNALS):
        return MODELS["haiku"]

    # General-purpose default for queries with no strong signal
    return MODELS["sonnet"]


def call_model(prompt: str, has_large_attachment: bool = False) -> str:
    model_id = route_query(prompt, has_large_attachment)

    response = client.messages.create(
        model=model_id,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )

    tier = next(k for k, v in MODELS.items() if v == model_id)
    print(f"[Router] → {tier.upper()} ({model_id})")
    return response.content[0].text

# ── 2. SEMANTIC CACHE ─────────────────────────────────────────────────────────
#
# If 500 employees ask the HR chatbot the same policy question today,
# the LLM should be called once — not 500 times.
#
# This implementation uses an exact-match hash cache (suitable for
# demonstrating the pattern). For production, replace the dict store
# with Redis and the hash with a vector similarity check so near-identical
# phrasings of the same question also result in a cache hit.

_cache: dict[str, dict] = {}
CACHE_TTL = 3600  # seconds — tune to your data freshness requirements


def _cache_key(prompt: str) -> str:
    # Normalise before hashing so minor whitespace/capitalisation differences
    # don't produce unnecessary cache misses on identical queries
    return hashlib.sha256(prompt.strip().lower().encode()).hexdigest()


def cached_call(prompt: str, has_large_attachment: bool = False) -> str:
    """
    Semantic cache wrapper around call_model().
    Returns a cached response when available; calls the model only on misses.
    """
    key = _cache_key(prompt)
    now = time.time()

    if key in _cache and (now - _cache[key]["ts"]) < CACHE_TTL:
        print("[Cache] HIT  — LLM call skipped, tokens saved.")
        return _cache[key]["response"]

    response = call_model(prompt, has_large_attachment)
    _cache[key] = {"response": response, "ts": now}
    print("[Cache] MISS — response stored for future hits.")
    return response


# ── DEMO ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # Simple extraction → routes to Haiku ($1/$5 per 1M tokens)
    print("\n--- Query 1: simple extraction ---")
    cached_call("Extract all dates and payment amounts from this invoice.")

    # Complex regulatory reasoning → routes to Opus ($5/$25 per 1M tokens)
    print("\n--- Query 2: complex regulatory task ---")
    cached_call(
        "Identify the regulatory gaps between our current AML policy "
        "and the latest FinCEN guidance on beneficial ownership."
    )

    # Same query as Query 1 — cache hit, no LLM call fired
    print("\n--- Query 3: repeated query (cache hit expected) ---")
    cached_call("Extract all dates and payment amounts from this invoice.")

    # Expected output:
    # --- Query 1: simple extraction ---
    # [Router] → HAIKU (claude-haiku-4-5)
    # [Cache] MISS — response stored for future hits.
    #
    # --- Query 2: complex regulatory task ---
    # [Router] → OPUS (claude-opus-4-8)
    # [Cache] MISS — response stored for future hits.
    #
    # --- Query 3: repeated query (cache hit expected) ---
    # [Cache] HIT  — LLM call skipped, tokens saved.
