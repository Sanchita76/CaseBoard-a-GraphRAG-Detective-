"""CaseBoard GraphRAG - run everything:  python run_demo.py   (add --no-llm for a fast run)"""
import argparse
import os

from graphrag import GraphRAG, NaiveRAG, build_graph, detect_communities, summarize_communities
from graphrag.query import format_result
from graphrag.viz import draw_graph

DEMO_QUESTIONS = [
    "How is Felix Ortega connected to the Blue Heron Pearl?",
    "How is Silas Crane connected to the Marlow Museum?",
    "Tell me about Nils Arden.",
    "What are the main groups in these case files?",
    "Who could have disabled the alarm with a maintenance code?",
]
# <<< Add your own questions below (use names from data/entities.json) >>>
MY_QUESTIONS = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="skip the language model (faster)")
    ap.add_argument("--model", default="google/flan-t5-base", help="HuggingFace model name")
    ap.add_argument("--question", help="ask one custom question")
    args = ap.parse_args()
    os.makedirs("outputs", exist_ok=True)

    print("1/4 Building knowledge graph ...")
    G, sentences, finder = build_graph("data")
    comms = detect_communities(G)
    summaries = summarize_communities(G, comms, sentences)
    print(f"    {G.number_of_nodes()} entities, {G.number_of_edges()} relationships, {len(comms)} communities")

    print("2/4 Saving graph picture + community report ...")
    draw_graph(G, out_file="outputs/graph.png")
    with open("outputs/communities.md", "w", encoding="utf-8") as f:
        f.write("# Communities found in the case files\n\n")
        for s in summaries:
            f.write(f"## Group {s['id'] + 1}: {s['title']}\n**Members:** {', '.join(s['members'])}\n\n")
            for k in s["key_sentences"]:
                f.write(f"- {k}\n")
            f.write("\n")

    llm = None
    if not args.no_llm:
        print("3/4 Loading free local language model (first time downloads ~1GB) ...")
        from graphrag.llm import load_llm
        llm = load_llm(args.model)
    else:
        print("3/4 Skipping language model (--no-llm)")

    print("4/4 Answering questions ...\n")
    engine = GraphRAG(G, sentences, summaries, finder, llm)
    questions = [args.question] if args.question else DEMO_QUESTIONS + MY_QUESTIONS
    report = ["# Answers\n"]
    for i, q in enumerate(questions):
        r = engine.answer(q)
        text = format_result(r)
        print(text + "\n" + "-" * 70)
        report.append("```\n" + text + "\n```\n")
        if r["path"]:
            draw_graph(G, path=r["path"], out_file=f"outputs/path_q{i + 1}.png",
                       title=f"Reasoning path: {q}")
    with open("outputs/answers.md", "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    # GraphRAG vs plain RAG on the multi-hop question
    q = "How is Silas Crane connected to the Marlow Museum?"
    naive = NaiveRAG(sentences).search(q, k=3)
    graph = engine.retrieve(q)
    with open("outputs/compare_naive_vs_graphrag.md", "w", encoding="utf-8") as f:
        f.write(f"# Plain RAG vs GraphRAG\n\nQuestion: {q}\n\n## Plain RAG (top-3 similar sentences)\n")
        f.writelines(f"- [{s['doc']}] {s['text']}\n" for s in naive)
        f.write("\n## GraphRAG (follows the connections)\n")
        f.writelines(f"- {x}\n" for x in graph["facts"])
    print("Done! Open the 'outputs' folder: graph.png, path_q*.png, answers.md, communities.md, compare_naive_vs_graphrag.md")


if __name__ == "__main__":
    main()
