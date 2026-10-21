
def recall_at_k(
    ranked_ids: list[str],
    relevant_ids: list[str],
    k: int,
) -> float:
    """Measure the fraction of relevant chunks found in the top k."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    retrieved = set(ranked_ids[:k])
    relevant = set(relevant_ids)

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(
    ranked_ids: list[str],
    relevant_ids: list[str],
) -> float:
    """Return the reciprocal rank of the first relevant result."""
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    ranked_lists: list[list[str]],
    relevant_lists: list[list[str]],
) -> float:
    """Average reciprocal rank across benchmark questions."""
    if len(ranked_lists) != len(relevant_lists):
        raise ValueError(
            "ranked_lists and relevant_lists must have equal lengths"
        )

    if not ranked_lists:
        raise ValueError("At least one ranked list is required")

    scores = [
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(ranked_lists, relevant_lists)
    ]

    return sum(scores) / len(scores)
