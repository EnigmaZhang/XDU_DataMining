import matplotlib.pyplot as plt
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import numpy as np

import math
from collections import OrderedDict

"""
@Author: Zhang Xiaotian
@School: Xidian University
@Reference:
https://people.maths.ox.ac.uk/porterm/research/mariano_thesis_final.pdf
This program is to build a bipartite network and do link prediction on the Movielens datasets.
"""

TEST_SIZE = 50
SEARCH_NUM = 500

"""
Read the data from a directory and change it into a list of (UserID, MovieID, rating).
"""


def read_data(path):
    rating_data = []
    rating_path = path + '/ratings.dat'
    with open(rating_path, 'r') as rating_file:
        for line in rating_file:
            # Use tuple to protect data from changing.
            rating_data.append(tuple(line.split("::")[:-1]))
    return rating_data


"""
Split data into train set and test set.
"""


def data_split(data):
    train, test = train_test_split(data, test_size=TEST_SIZE, shuffle=True)
    return train, test


"""
Build a bipartite network from data.
"""


def build_network(data):
    B = nx.Graph()
    B.add_nodes_from(["U" + i[0] for i in data], bipartite=0)
    B.add_nodes_from(["M" + i[1] for i in data], bipartite=1)
    B.add_weighted_edges_from([("U" + i[0], "M" + i[1], int(i[2])) for i in data])
    return B


"""
Compute with matrix to get node similarities.
"""


def get_node_similarities(user, net):
    # Get all reviewed movies of user.
    similarities = {}
    user_node = [i[1] for i in net.edges(user)]
    for user_edge in net.edges(user):
        # Get users who review the same movie as user.
        for movie_edge in net.edges(user_edge[1]):
            another_user = movie_edge[1]
            if user == another_user:
                continue
            if another_user in similarities:
                continue
            # Get reviewed movie of one user.
            movie_node = [i[1] for i in net.edges(another_user)]
            common_movies = set(user_node) & set(movie_node)
            # common_movies = set(user_node)
            multiply = 0
            user_square = 0
            another_square = 0
            for movie in common_movies:
                user_weight = net.get_edge_data(user, movie)["weight"]
                # if not net.has_edge(another_user, movie):
                #     another_weight = 0
                # else:
                another_weight = net.get_edge_data(another_user, movie)["weight"]
                multiply += user_weight * another_weight
                user_square += pow(user_weight, 2)
                another_square += pow(another_weight, 2)
            cosine_similarity = multiply / (math.sqrt(user_square) * math.sqrt(another_square))
            similarities[another_user] = cosine_similarity
    return similarities


"""
Get prediction from the most similar user.
"""


def get_prediction(user, movie, net):
    user = "U" + user
    movie = "M" + movie
    similarities = get_node_similarities(user, net)
    items = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    new_dict = OrderedDict(items)
    keys = list(new_dict.keys())
    # mode
    weights = []
    times = 0
    for k in keys:
        if net.has_edge(k, movie):
            weights.append(net.get_edge_data(k, movie)["weight"])
        times += 1
        if times > SEARCH_NUM:
            break
    if len(weights) > 0:
        counts = np.bincount(weights)
        star = np.argmax(counts)
    else:
        star = 2
    return star >= 3


if __name__ == "__main__":
    processed_data = read_data("./data/")
    train, test = data_split(processed_data)
    BNet = build_network(train)
    prediction = [get_prediction(edge[0], edge[1], BNet) for edge in test]
    real_data = [int(edge[2]) >= 3 for edge in test]
    print(prediction)
    print(real_data)
    print(accuracy_score(real_data, prediction))
    fpr, tpr, thresholds = roc_curve(real_data, prediction, pos_label=True)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()


