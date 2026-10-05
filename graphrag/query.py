"""Step 3 of GraphRAG: answer questions by walking the graph.

Three retrieval modes (chosen automatically):
  link     - question mentions 2+ entities  -> find the connection path between them (multi-hop)
  profile  - question mentions 1 entity     -> collect that entity's neighbourhood
  global   - big-picture question           -> use the community summaries
  keyword  - nothing recognised             -> fall back to TF-IDF sentence search
"""
import re

import networkx as nx

from .baseline import NaiveRAG

GLOBAL_WORDS = re.compile(
    r"\b(overall|main groups?|themes?|clusters?|big picture|summar\w+|communities|groups)\b", re.I
)


class GraphRAG:
    def __init__(self, G, sentences, summaries, finder, llm=None):
        self.G, self.sentences, self.summaries, self.finder, self.llm = (
            G, sentences, summaries, finder, llm,
        )
        self.fallback = NaiveRAG(sentences)

    # ---------- retrieval ----------
    def _edge_evidence(self, a, b, limit=2):
        ids = self.G[a][b]["evidence"][:limit]
        return [self.sentences[i] for i in ids]

    def retrieve(self, question):
        ents = [e for e in self.finder.find(question) if e in self.G]
        is_global = bool(GLOBAL_WORDS.search(question))

        if len(ents) >= 2:
            return self._link_mode(ents)
        if len(ents) == 1:
            return self._profile_mode(ents[0])
        if is_global:
            return self._global_mode()
        hits = self.fallback.search(question, k=4)
        return {"mode": "keyword", "entities": [], "path": [], "evidence": hits,
                "facts": [h["text"] for h in hits]}

    def _link_mode(self, ents):
        full_path, evidence = [], []
        for a, b in zip(ents[:-1], ents[1:]):
            if nx.has_path(self.G, a, b):
                p = nx.shortest_path(self.G, a, b)
            else:
                p = [a, b]
            full_path += p if not full_path else p[1:]
            for u, v in zip(p[:-1], p[1:]):
                if self.G.has_edge(u, v):
                    evidence += self._edge_evidence(u, v)
        seen, uniq = set(), []
        for s in evidence:
            if s["id"] not in seen:
                seen.add(s["id"])
                uniq.append(s)
        chain = " -> ".join(full_path)
        hops = max(len(full_path) - 1, 0)
        facts = [f"Connection chain ({hops} hop{'s' if hops != 1 else ''}): {chain}."]
        facts += [s["text"] for s in uniq]
        return {"mode": "link", "entities": ents, "path": full_path, "evidence": uniq, "facts": facts}

    def _profile_mode(self, ent):
        G = self.G
        neighbours = sorted(G[ent], key=lambda n: -G[ent][n]["weight"])
        mine = [s for s in self.sentences if ent in s["entities"]]
        mine.sort(key=lambda s: -len(s["entities"]))
        mine = mine[:6]
        group = self.summaries[G.nodes[ent]["community"]]
        facts = [f"{ent} is a {G.nodes[ent]['type'].lower()} directly linked to: {', '.join(neighbours)}."]
        facts += [s["text"] for s in mine]
        return {"mode": "profile", "entities": [ent], "path": [], "evidence": mine,
                "facts": facts, "group": group}

    def _global_mode(self):
        facts = []
        for s in self.summaries:
            if len(s["members"]) < 2:
                continue
            facts.append(f"Group {s['id'] + 1} ({', '.join(s['members'])}): " + " ".join(s["key_sentences"][:2]))
        return {"mode": "global", "entities": [], "path": [], "evidence": [], "facts": facts}

    # ---------- answering ----------
    def _template_answer(self, r):
        if r["mode"] == "link":
            return r["facts"][0] + " Evidence is listed below."
        if r["mode"] == "profile":
            return r["facts"][0]
        if r["mode"] == "global":
            return "The case files split into these groups:\n" + "\n".join("  - " + f for f in r["facts"])
        return "Closest matching sentences found (no entity recognised in the question)."

    def answer(self, question):
        r = self.retrieve(question)
        r["question"] = question
        r["template_answer"] = self._template_answer(r)
        if self.llm:
            context = "\n".join("- " + f for f in r["facts"])
            prompt = (
                "Answer the question using only the facts below. Be specific and name the "
                "people and organisations involved.\n\nFacts:\n" + context +
                f"\n\nQuestion: {question}\nAnswer:"
            )
            r["llm_answer"] = self.llm(prompt)
        return r


def format_result(r):
    lines = [f"Q: {r['question']}", f"Mode: {r['mode']}"]
    if r["path"]:
        lines.append("Path: " + " -> ".join(r["path"]))
    lines.append("Answer: " + r.get("llm_answer", r["template_answer"]))
    if r.get("llm_answer"):
        lines.append("Graph facts: " + r["template_answer"])
    if r["evidence"]:
        lines.append("Evidence:")
        lines += [f"  [{e['doc']}] {e['text']}" for e in r["evidence"]]
    return "\n".join(lines)
