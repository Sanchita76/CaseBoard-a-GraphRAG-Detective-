"""Step 1 of GraphRAG: turn raw text files into a knowledge graph.

Nodes  = entities (people, organisations, places, items)
Edges  = "these two entities appear in the same sentence"
         (edge keeps the sentences as evidence + the main verb as a label)
"""
import glob
import json
import os
import re
from itertools import combinations

import networkx as nx
import spacy

PRONOUN_STARTS = ("he ", "she ", "his ", "her ", "they ", "their ")


class EntityFinder:
    """Finds known entities (and their aliases) inside a piece of text."""

    def __init__(self, entities):
        self.type_of = {e["name"]: e["type"] for e in entities}
        self.alias_to_name = {}
        for e in entities:
            for alias in [e["name"], *e.get("aliases", [])]:
                self.alias_to_name[alias.lower()] = e["name"]
        # longest aliases first so "Crane Auctions" wins over "Crane"
        aliases = sorted(self.alias_to_name, key=len, reverse=True)
        self.pattern = re.compile(
            r"\b(" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE
        )

    def find(self, text):
        found, seen = [], set()
        for m in self.pattern.finditer(text):
            name = self.alias_to_name[m.group(1).lower()]
            if name not in seen:
                seen.add(name)
                found.append(name)
        return found


def load_entities(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_documents(folder):
    docs = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def build_graph(data_dir="data", nlp=None):
    """Returns (graph, sentences, finder).

    sentences = list of dicts: {id, doc, text, entities}
    """
    entities = load_entities(os.path.join(data_dir, "entities.json"))
    finder = EntityFinder(entities)
    nlp = nlp or spacy.load("en_core_web_sm", disable=["ner"])

    G = nx.Graph()
    for e in entities:
        G.add_node(e["name"], type=e["type"], mentions=0)
    sentences = []

    for fname, text in read_documents(os.path.join(data_dir, "case_files")):
        prev_people = []
        for sent in nlp(text).sents:
            s = sent.text.strip()
            if not s:
                continue
            ents = finder.find(s)
            people = [e for e in ents if finder.type_of[e] == "PERSON"]

            # light "he/she" resolution: only when the previous sentence had exactly one person
            if not people and s.lower().startswith(PRONOUN_STARTS) and len(prev_people) == 1:
                ents.insert(0, prev_people[0])
                people = [prev_people[0]]
            prev_people = people

            root = next((t.lemma_.lower() for t in sent if t.dep_ == "ROOT"), "relates to")
            sid = len(sentences)
            sentences.append({"id": sid, "doc": fname, "text": s, "entities": ents, "verb": root})

            for e in ents:
                G.nodes[e]["mentions"] += 1
            for a, b in combinations(sorted(set(ents)), 2):
                if not G.has_edge(a, b):
                    G.add_edge(a, b, weight=0, relations=[], evidence=[])
                G[a][b]["weight"] += 1
                G[a][b]["relations"].append(root)
                G[a][b]["evidence"].append(sid)

    G.remove_nodes_from([n for n in list(G.nodes) if G.nodes[n]["mentions"] == 0])
    return G, sentences, finder
