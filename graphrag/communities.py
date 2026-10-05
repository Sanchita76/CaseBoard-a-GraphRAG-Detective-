"""Step 2 of GraphRAG: find communities (groups) in the graph and summarise each one.

This is the 'Graph' idea from Microsoft's GraphRAG: instead of only searching single
sentences, we also keep a summary of every cluster so we can answer big-picture questions.
Summaries here are extractive (no LLM needed), so this step is fast and free.
"""
import networkx as nx


def detect_communities(G, seed=42):
    comms = nx.algorithms.community.louvain_communities(G, weight="weight", seed=seed)
    comms = sorted(comms, key=lambda c: (-len(c), sorted(c)[0]))
    for i, c in enumerate(comms):
        for n in c:
            G.nodes[n]["community"] = i
    return comms


def summarize_communities(G, comms, sentences, max_sentences=3):
    summaries = []
    for i, c in enumerate(comms):
        members = sorted(c, key=lambda n: (-G.degree(n, weight="weight"), n))
        scored = []
        for s in sentences:
            inside = [e for e in s["entities"] if e in c]
            # keep sentences that talk mostly about this group
            if len(inside) >= 2 or (len(inside) == 1 and len(s["entities"]) == 1):
                scored.append((len(inside), -s["id"], s))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        key_sentences = [s["text"] for _, _, s in scored[:max_sentences]]
        summaries.append(
            {
                "id": i,
                "title": " / ".join(members[:2]),
                "members": members,
                "key_sentences": key_sentences,
            }
        )
    return summaries
