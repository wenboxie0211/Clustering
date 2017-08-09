def Q(graph, cluster):
    e = 0.0
    a_2 = 0.0
    cluster_degree_table = {}
    for vtx, adj in graph.edge.iteritems():
        label = cluster[vtx]
        for neighbor in adj.keys():
            if label == cluster[neighbor]:
                e += 1
        if label not in cluster_degree_table:
            cluster_degree_table[label] =0
        cluster_degree_table[label] += len(adj)
    e /= 2 * graph.number_of_edges()

    for label, cnt in cluster_degree_table.iteritems():
        a = 0.5 * cnt / graph.number_of_edges()
        a_2 += a * a

    Q = e - a_2
    return Q