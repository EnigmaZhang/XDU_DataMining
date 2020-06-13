import networkx as nx
import matplotlib.pyplot as plt

"""
@Author: Zhang Xiaotian
@School: Xidian University
@Reference:
https://sikasjc.github.io/2017/12/20/GN/
This program is to use GN algorithm to find community.
"""


class GN:
    def __init__(self, path):
        self.G = nx.read_gml(path, label='id')
        self.G_copy = self.G.copy()
        self.partition = [[n for n in self.G.nodes()]]
        self.all_Q = [0.0]
        self.max_Q = 0.0

    # Using max_Q to divide communities
    def run(self):
        # Until there is no edge in the graph
        while len(self.G.edges()) != 0:
            # Find the most betweenness edge
            edge = max(nx.edge_betweenness_centrality(self.G).items(), key=lambda item: item[1])[0]
            # Remove the most betweenness edge
            self.G.remove_edge(edge[0], edge[1])
            # Get the the connected nodes
            components = [list(c) for c in list(nx.connected_components(self.G))]
            # When the dividing is needed, this is for finding the maxQ and record it while trying.
            if len(components) != len(self.partition):
                # Compute Q
                currentQ = self.calculateQ(components, self.G_copy)
                if currentQ not in self.all_Q:
                    self.all_Q.append(currentQ)
                if currentQ > self.max_Q:
                    self.max_Q = currentQ
                    self.partition = components

        print('The number of communities:', len(self.partition))
        print('Max_Q:', self.max_Q)
        print(self.partition)

    # Divide the graph into n parts.
    def run_n(self, n):
        # Until there is no edge in the graph
        while len(self.G.edges()) != 0:
            # Find the most betweenness edge
            edge = max(nx.edge_betweenness_centrality(self.G).items(), key=lambda item: item[1])[0]
            # Remove the most betweenness edge
            self.G.remove_edge(edge[0], edge[1])
            # Get the the connected nodes
            components = [list(c) for c in list(nx.connected_components(self.G))]
            # Divide the graph into n parts.
            if len(components) <= n:
                # Compute Q
                currentQ = self.calculateQ(components, self.G_copy)
                if currentQ not in self.all_Q:
                    self.all_Q.append(currentQ)
                if currentQ > self.max_Q:
                    self.max_Q = currentQ
                    self.partition = components

        print('The number of communities:', len(self.partition))
        print('Max_Q:', self.max_Q)
        print(self.partition)
        return self.partition, self.all_Q, self.max_Q

    # Drawing Q when dividing.
    def draw_Q(self):
        plt.plot(self.all_Q)
        plt.show()

    # Computing the Q
    @staticmethod
    def calculateQ(partition, G):
        """
        可以用模块度函数Q定量描述社区划分的模块化水平。假设已经发现复杂网络的社区结构，M为已发现的社区个数，L为网络中的边数，
        ls是社区 s中节点相互连接的数目，ds是社区s中所有节点相互连接数目的和。
        """
        L = len(G.edges())
        Q = 0.0

        for community in partition:
            ds = 0
            for node in community:
                ds += len([x for x in G.neighbors(node)])
            ls = 0
            for i in range(len(community)):
                for j in range(len(community)):
                    if G.has_edge(community[i], community[j]):
                        ls += 1
            # One edge is counted twice.
            ls /= 2
            Q += ls / L - pow(ds / (2 * L), 2)

        return Q

    def add_group(self):
        num = 0
        nodegroup = {}
        for partition in self.partition:
            for node in partition:
                nodegroup[node] = {'group': num}
            num = num + 1
        print(nodegroup)
        nx.set_node_attributes(self.G_copy, nodegroup)

    def to_gml(self, path):
        nx.write_gml(self.G_copy, path)


if __name__ == '__main__':
    net = GN(r'.\data\karate.gml')
    net.run()
    net.draw_Q()
    net.add_group()
    net.to_gml(r'.\data\out.gml')

    net = GN(r'.\data\karate.gml')
    net.run_n(2)
    net.draw_Q()
    net.add_group()
    net.to_gml(r'.\data\two_parts.gml')
