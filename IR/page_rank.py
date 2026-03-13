import numpy as np

def page_rank(graph, damping_factor=0.85, max_iterations=100):
    num_nodes = len(graph)
    # Start with equal rank for all pages [cite: 393]
    page_ranks = np.ones(num_nodes) / num_nodes

    for _ in range(max_iterations):
        prev_ranks = np.copy(page_ranks)
        for node in range(num_nodes):
            # Find pages that link TO the current node [cite: 397]
            incoming_links = [i for i, links in enumerate(graph) if node in links]

            # Sum the rank of incoming pages divided by their total out-links [cite: 400, 402]
            rank_sum = sum(prev_ranks[link] / len(graph[link]) for link in incoming_links)

            # Apply the PageRank formula with damping factor [cite: 399]
            page_ranks[node] = (1 - damping_factor) / num_nodes + damping_factor * rank_sum

        # Stop if the values have stabilized [cite: 404]
        if np.linalg.norm(page_ranks - prev_ranks, 2) < 1e-6:
            break
    return page_ranks

# Graph: A->[B, C], B->[C], C->[A]
web_graph = [[1, 2], [2], [0]]
results = page_rank(web_graph)

pages = ['Page A', 'Page B', 'Page C']
for i, rank in enumerate(results):
    print(f"{pages[i]}: {round(rank, 4)}")


# ----

import numpy as np

def calculate_pagerank(graph, d=0.85, iters=100):
    n = len(graph)
    # Initialize ranks equally [cite: 393]
    ranks = np.ones(n) / n

    for _ in range(iters):
        new_ranks = np.zeros(n)
        for i in range(n):
            # Find all pages 'j' that link to 'i' [cite: 397]
            incoming = [j for j, links in enumerate(graph) if i in links]

            # Sum the (Rank of j / Out-degree of j) [cite: 400, 402]
            rank_sum = sum(ranks[j] / len(graph[j]) for j in incoming if len(graph[j]) > 0)

            # Apply PageRank formula
            new_ranks[i] = (1 - d) / n + d * rank_sum

        if np.linalg.norm(new_ranks - ranks) < 1e-6: # Check convergence [cite: 404]
            break
        ranks = new_ranks
    return ranks

# Representing the graph: A->[B,C,D], B->[C,E], C->[A,D], D->[], E->[]
web_graph = [[1, 2, 3], [2, 4], [0, 3], [], []]
final_ranks = calculate_pagerank(web_graph)

pages = ['Page A', 'Page B', 'Page C', 'Page D', 'Page E']
for page, rank in zip(pages, final_ranks):
    print(f"{page}: {round(rank, 4)}")


# ----
import numpy as np

def page_rank(graph, damping_factor=0.85, max_iterations=100):
    num_nodes = len(graph)
    # Start with equal rank for all pages [cite: 393]
    page_ranks = np.ones(num_nodes) / num_nodes

    for _ in range(max_iterations):
        prev_page_ranks = np.copy(page_ranks)
        for node in range(num_nodes):
            # Find pages that link TO the current node [cite: 397]
            incoming_links = [i for i, links in enumerate(graph) if node in links]

            # Sum the rank of incoming pages divided by their out-degree [cite: 402, 403]
            rank_sum = sum(prev_page_ranks[link] / len(graph[link]) for link in incoming_links)

            # Apply the PageRank formula with damping factor [cite: 399]
            page_ranks[node] = (1 - damping_factor) / num_nodes + damping_factor * rank_sum

        # Stop if values stabilize [cite: 404, 405]
        if np.linalg.norm(page_ranks - prev_page_ranks, 2) < 1e-6:
            break
    return page_ranks

# Representing the graph indices: A=0, B=1, C=2, D=3
web_graph = [[1, 2], [2, 3], [0, 3], [1]]
result = page_rank(web_graph)

pages = ['Page A', 'Page B', 'Page C', 'Page D']
for i, rank in enumerate(result):
    print(f"{pages[i]}: {round(rank, 4)}")
