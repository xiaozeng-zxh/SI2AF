from args import args
import pickle as pkl
import numpy as np
import random
from sip import *
import networkx as nx
import sys

with open(f"./Detector/{args.dataset[:3]}_user_news_mapping.pkl", "rb") as file:
    user_news_mapping = pkl.load(file)

def user_selection(acc_list, height):
    acc_list = list(random.sample(acc_list, 170))
    num_node = len(acc_list)
    adj_matrix = np.zeros((num_node, num_node))
    for acc_id1, acc1 in enumerate(acc_list):
        news = user_news_mapping[acc1]
        for acc_id2, acc2 in enumerate(acc_list):
            if acc_id1 == acc_id2:
                continue
            adj_matrix[acc_id1, acc_id2] += 1.0

    for id in range(num_node):
        if sum(adj_matrix[id]) == 0:
            adj_matrix[id] += 1.0 / num_node
        else:
            adj_matrix[id] /= sum(adj_matrix[id])

    y = PartitionTree(adj_matrix=adj_matrix)
    x = y.build_encoding_tree(height)

    pes = []
    for nid in range(num_node):
        pes.append(y.path_entropy(nid, 0.0))
    pes = np.array(pes)
    sorted_indices = np.argsort(pes)
    indices = [sorted_indices[:100], sorted_indices[100:150], sorted_indices[150:]]
    controlled_user = []
    for indice in indices:
        user_list = []
        for nid in indice:
            user_list.append(acc_list[nid])
        controlled_user.append(user_list)
    return controlled_user
