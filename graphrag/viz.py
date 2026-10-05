"""Draw the knowledge graph (coloured by community) and optionally highlight a path."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2", "#9D755D"]


def draw_graph(G, path=None, out_file="outputs/graph.png", title="Case knowledge graph"):
    pos = nx.spring_layout(G, seed=7, k=0.9, weight="weight")
    plt.figure(figsize=(13, 9))
    colors = [PALETTE[G.nodes[n].get("community", 0) % len(PALETTE)] for n in G]
    sizes = [500 + 250 * G.degree(n) for n in G]
    widths = [0.8 + 0.9 * G[u][v]["weight"] for u, v in G.edges]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.35, edge_color="#555555")
    if path and len(path) > 1:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=4.5, edge_color="#D62728")
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors="white")
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold")
    plt.title(title, fontsize=15)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_file, dpi=140)
    plt.close()
    return out_file
