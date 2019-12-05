from pyvis.network import Network
import torch_geometric
import networkx as nx
from operator import itemgetter

def show(data):
    G = Network(notebook=True)

    node_indexes = range(data.num_nodes)
    node_y = [int(y) for y in data.y]
    colors = ['#ff0000' if i else '#0000ff' for i in node_y]

    G.add_nodes(node_indexes, color=colors)

    for i, j in data.edge_index.t().numpy():
        G.add_edge(int(i), int(j))

    G.show('example.html')

def show_ego_graph(data, index):
    # convert to graph
    G = torch_geometric.utils.to_networkx(
        dataset.data, node_attrs=None, edge_attrs=None
    )

    # Create ego graph of main hub
    hub_ego = nx.ego_graph(G, index)

    # Draw graph
    pos = nx.spring_layout(hub_ego)
    nx.draw(hub_ego, pos, node_color='b', node_size=50, with_labels=False)

    # Draw ego as large and red
    nx.draw_networkx_nodes(hub_ego, pos, nodelist=[index], node_size=300, node_color='r')
    plt.show()
    hub_ego = nx.ego_graph(G, largest_hub)