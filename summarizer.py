import re

import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def textrank_summarize(article, num_sentences=3):
    """
    LAB7 ka exact TextRank pipeline, saath me har step ka trace return.
    Returns:
        {
          "summary": str,
          "steps": [ {step, title, description, type, data}, ... ],
          "num_sentences": int,
        }
    """
    steps = []

    # ---------------- Step 1: Sentence splitting ----------------
    sentences = re.split(r'(?<=[.!?])\s+', article.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    steps.append({
        "step": 1,
        "title": "Step 1 — Split Article into Sentences",
        "description": "re.split(r'(?<=[.!?])\\s+') se text ko sentences me toda, empty strings remove kiye.",
        "type": "sentences",
        "data": [{"index": i, "text": s} for i, s in enumerate(sentences)],
    })

    if not sentences:
        return {"summary": "No article text provided.", "steps": steps, "num_sentences": 0}

    if len(sentences) <= num_sentences:
        summary = " ".join(sentences)
        steps.append({
            "step": 8,
            "title": "Step 8 — Return Sentences in Original Order",
            "description": "Sentences count <= requested; isliye saare sentences summary hain.",
            "type": "summary",
            "data": summary,
        })
        return {"summary": summary, "steps": steps, "num_sentences": len(sentences)}

    # ---------------- Step 2: TF-IDF vectors ----------------
    vectorizer = TfidfVectorizer(stop_words="english")
    sentence_vectors = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out().tolist()

    rows = []
    for i in range(sentence_vectors.shape[0]):
        row = sentence_vectors.getrow(i)
        entries = [
            {"word": feature_names[j], "weight": round(float(val), 4)}
            for j, val in zip(row.indices, row.data)
        ]
        entries.sort(key=lambda x: x["weight"], reverse=True)
        rows.append({
            "index": i,
            "sentence": sentences[i],
            "features": entries,
        })

    steps.append({
        "step": 2,
        "title": "Step 2 — TF-IDF Vectorization",
        "description": (
            f"TfidfVectorizer(stop_words='english') se har sentence ka TF-IDF vector banaya. "
            f"Vocabulary size = {len(feature_names)}, Matrix shape = {sentence_vectors.shape}"
        ),
        "type": "vectors",
        "data": {
            "vocab_size": len(feature_names),
            "shape": [int(sentence_vectors.shape[0]), int(sentence_vectors.shape[1])],
            "rows": rows,
        },
    })

    # ---------------- Step 3: Cosine similarity ----------------
    similarity_matrix = cosine_similarity(sentence_vectors)
    rounded3 = np.round(similarity_matrix, 3)

    steps.append({
        "step": 3,
        "title": "Step 3 — Cosine Similarity Matrix",
        "description": "cosine_similarity() se har sentence pair ke beech similarity nikali (range 0–1).",
        "type": "matrix",
        "data": {
            "matrix": rounded3.tolist(),
            "sentences": [{"index": i, "preview": s[:60]} for i, s in enumerate(sentences)],
        },
    })

    # ---------------- Step 4: Remove self-connections ----------------
    np.fill_diagonal(similarity_matrix, 0)
    rounded4 = np.round(similarity_matrix, 3)

    steps.append({
        "step": 4,
        "title": "Step 4 — Remove Self-Connections",
        "description": "np.fill_diagonal(similarity_matrix, 0) se diagonal (self-similarity) 0 ki.",
        "type": "matrix",
        "data": {
            "matrix": rounded4.tolist(),
            "sentences": [{"index": i, "preview": s[:60]} for i, s in enumerate(sentences)],
        },
    })

    # ---------------- Step 5: Weighted graph ----------------
    graph = nx.from_numpy_array(similarity_matrix)
    edges = [
        {"u": int(u), "v": int(v), "weight": round(float(d.get("weight", 0)), 3)}
        for u, v, d in graph.edges(data=True)
    ]

    steps.append({
        "step": 5,
        "title": "Step 5 — Create Weighted Graph",
        "description": (
            f"nx.from_numpy_array() se undirected weighted graph banaya. "
            f"Nodes = {graph.number_of_nodes()}, Edges = {graph.number_of_edges()}"
        ),
        "type": "graph",
        "data": {
            "nodes": [int(n) for n in graph.nodes()],
            "edges": edges,
        },
    })

    # ---------------- Step 6: PageRank ----------------
    scores = nx.pagerank(graph, weight="weight")
    score_list = [
        {
            "index": int(i),
            "sentence": sentences[i],
            "score": round(float(s), 4),
        }
        for i, s in scores.items()
    ]
    score_list_sorted = sorted(score_list, key=lambda x: x["score"], reverse=True)

    steps.append({
        "step": 6,
        "title": "Step 6 — Apply PageRank",
        "description": "nx.pagerank(graph, weight='weight') se har sentence ka importance score nikala.",
        "type": "scores",
        "data": score_list_sorted,
    })

    # ---------------- Step 7: Select top-ranked ----------------
    ranked_sentences = sorted(scores, key=scores.get, reverse=True)
    top_n = min(num_sentences, len(sentences))
    selected_indices = sorted(ranked_sentences[:top_n])

    steps.append({
        "step": 7,
        "title": "Step 7 — Select Top-Ranked Sentences",
        "description": (
            f"PageRank scores ke hisaab se top {top_n} sentences chuni, "
            f"phir original order me sort kiya."
        ),
        "type": "selected",
        "data": {
            "ranked": [int(i) for i in ranked_sentences],
            "top": [int(i) for i in ranked_sentences[:top_n]],
            "selected_original_order": [int(i) for i in selected_indices],
            "sentences": [{"index": i, "text": sentences[i]} for i in range(len(sentences))],
        },
    })

    # ---------------- Step 8: Original-order summary ----------------
    summary = " ".join(sentences[i] for i in selected_indices)

    steps.append({
        "step": 8,
        "title": "Step 8 — Return Sentences in Original Order",
        "description": "Selected sentences ko original order me join karke final summary banai.",
        "type": "summary",
        "data": summary,
    })

    return {"summary": summary, "steps": steps, "num_sentences": len(sentences)}
