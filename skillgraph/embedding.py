"""Embedding utilities for node similarity computation."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Sequence

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("skillgraph.embedding")


def embed_texts(texts: Sequence[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Embed a batch of texts using OpenAI's embedding API.

    Returns an (N, dim) numpy array of embedding vectors.
    """
    import openai

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set; falling back to local hash embeddings")
        return _embed_texts_local(texts)

    client = openai.OpenAI(api_key=api_key)
    try:
        resp = client.embeddings.create(model=model, input=list(texts))
    except Exception as exc:
        logger.warning("Embedding API failed (%s); falling back to local hash embeddings", exc)
        return _embed_texts_local(texts)

    vectors = [item.embedding for item in sorted(resp.data, key=lambda x: x.index)]
    logger.debug("Embedded %d texts (%d dims)", len(vectors), len(vectors[0]))
    return np.array(vectors, dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of vectors.

    Args:
        a: (M, dim) array
        b: (N, dim) array

    Returns:
        (M, N) similarity matrix
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def extract_node_embed_text(markdown: str) -> str:
    """Extract the text to embed from a node's markdown.

    Uses the YAML frontmatter description + triggers.
    """
    description = ""
    triggers = []

    # Extract description
    desc_match = re.search(r'^description:\s*"?(.+?)"?\s*$', markdown, re.MULTILINE)
    if desc_match:
        description = desc_match.group(1)

    # Extract triggers
    in_triggers = False
    for line in markdown.split("\n"):
        if line.strip() == "triggers:":
            in_triggers = True
            continue
        if in_triggers:
            t_match = re.match(r'^\s+-\s+"?(.+?)"?\s*$', line)
            if t_match:
                triggers.append(t_match.group(1))
            elif not line.strip().startswith("-"):
                break

    parts = [description] + triggers
    return " | ".join(p for p in parts if p)


def _embed_texts_local(texts: Sequence[str], dim: int = 1024) -> np.ndarray:
    """Deterministic local fallback embeddings using hashed token counts.

    This keeps the pipeline runnable when remote embedding APIs are unavailable.
    It is weaker than model-based embeddings but good enough for approximate
    task-local clustering.
    """
    matrix = np.zeros((len(texts), dim), dtype=np.float32)

    for row, text in enumerate(texts):
        tokens = _tokenize_for_local_embedding(text)
        if not tokens:
            continue
        for token in tokens:
            bucket = _stable_bucket(token, dim)
            matrix[row, bucket] += 1.0

    # L2 normalize so cosine similarity behaves as expected.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    matrix /= norms

    logger.info("Locally embedded %d texts into %d dims", len(texts), dim)
    return matrix


def _tokenize_for_local_embedding(text: str) -> list[str]:
    """Simple tokenizer for fallback lexical embeddings."""
    return re.findall(r"[a-z0-9<>_-]+", text.lower())


def _stable_bucket(token: str, dim: int) -> int:
    """Map a token deterministically to a vector bucket."""
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % dim


def find_top_k_matches(
    query_nodes: dict[str, str],
    candidate_nodes: dict[str, str],
    k: int = 3,
) -> dict[str, list[tuple[str, float]]]:
    """Find top-K most similar candidate nodes for each query node.

    Args:
        query_nodes: {node_id: markdown} for query subgraph
        candidate_nodes: {node_id: markdown} for candidate subgraph
        k: number of top matches to return

    Returns:
        {query_node_id: [(candidate_node_id, similarity_score), ...]}
        sorted by similarity descending.
    """
    if not query_nodes or not candidate_nodes:
        return {}

    q_ids = list(query_nodes.keys())
    c_ids = list(candidate_nodes.keys())

    q_texts = [extract_node_embed_text(query_nodes[nid]) for nid in q_ids]
    c_texts = [extract_node_embed_text(candidate_nodes[nid]) for nid in c_ids]

    logger.info("Embedding %d query nodes + %d candidate nodes", len(q_ids), len(c_ids))

    q_vecs = embed_texts(q_texts)
    c_vecs = embed_texts(c_texts)

    sim_matrix = cosine_similarity_matrix(q_vecs, c_vecs)

    results = {}
    for i, qid in enumerate(q_ids):
        sims = sim_matrix[i]
        top_indices = np.argsort(sims)[::-1][:k]
        results[qid] = [(c_ids[j], float(sims[j])) for j in top_indices]

        # Log for debugging
        matches_str = ", ".join(f"{c_ids[j]}({sims[j]:.3f})" for j in top_indices)
        logger.info("  %s → top-%d: [%s]", qid, k, matches_str)

    return results
