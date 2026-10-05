# How CaseBoard works (plain English)

## The problem with normal RAG
Normal RAG = "find the sentences that look most like the question, give them to an AI."
That fails for questions that need **several facts from different documents chained together**.
Example: *How is Silas Crane connected to the Marlow Museum?* No single sentence says it.
The answer needs: Crane is owed money by Ortega (file 03) → Ortega used to work at the museum (file 03/04).

## The GraphRAG idea
Build a **map of who/what is connected to whom**, then walk the map.

| Step | File | What happens |
|---|---|---|
| 1. Read | `build_graph.py` | spaCy splits text into sentences. Every sentence that mentions 2+ known entities creates/strengthens an edge between them. The sentence is stored as *evidence*; the sentence's main verb (e.g. "owns", "fire") labels the edge. |
| 2. Cluster | `communities.py` | The Louvain algorithm groups entities that are densely connected (a "community"). For each group we store its members and its most informative sentences = a mini report. |
| 3. Retrieve | `query.py` | Looks at the question: 2+ entities -> shortest path between them (multi-hop). 1 entity -> its neighbourhood. Big-picture words -> community reports. Otherwise -> TF-IDF search. |
| 4. Answer | `query.py` / `llm.py` | The collected facts are shown as evidence. Optionally FLAN-T5 (a small free model) rewrites them as a sentence. |
| 5. Show | `viz.py` | Draws the graph; the reasoning path is highlighted in red. |

## Glossary
- **Node**: one entity (a person, company...).
- **Edge**: a link between two nodes.
- **Community**: a cluster of nodes with many links among themselves.
- **Multi-hop**: an answer that needs 2+ links (A→B→C).
- **Evidence**: original sentences that justify a link, so answers can be verified.

## Ideas to extend
- Add more case files and entities.
- Replace the dictionary-based entity finder with spaCy NER (`PERSON`, `ORG`).
- Replace extractive group reports by LLM-written summaries (true Microsoft-style GraphRAG).
