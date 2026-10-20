import re


def normalize_text(text: str) -> str:
    """Normalize text for basic, case-insensitive evidence matching."""
    text = text.casefold()
    text = re.sub(r"[-\u2013\u2014]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def evaluate_groundedness(
    claims: list[str],
    retrieved_chunks: list[dict],
) -> dict:
    """Check normalized exact-text support; this does not assess entailment."""
    evidence = normalize_text(
        " ".join(chunk.get("text", "") for chunk in retrieved_chunks)
    )

    supported = []
    unsupported = []

    for claim in claims:
        normalized_claim = normalize_text(claim)

        if normalized_claim and normalized_claim in evidence:
            supported.append(claim)
        else:
            unsupported.append(claim)

    total = len(claims)
    score = len(supported) / total if total else 1.0

    return {
        "score": score,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "passed": score >= 0.75,
    }
