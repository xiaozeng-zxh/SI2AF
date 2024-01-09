from args import args
import pickle as pkl
import numpy as np
import random
from sip import *
import networkx as nx
import sys

with open(f"./Detector/{args.dataset[:3]}_news_user_mapping.pkl", "rb") as file:
    news_user_mapping = pkl.load(file)
with open(f"./Detector/{args.dataset[:3]}_user_news_mapping.pkl", "rb") as file:
    user_news_mapping = pkl.load(file)

def news_partition(target_news, controlled_user, news_prob_list, height):

    ctrled_user_list = sum([user for user in controlled_user], [])
    real_news = []
    for ctrled_user in ctrled_user_list:
        asso_news_list = user_news_mapping[ctrled_user]
        for news in asso_news_list:
            if (news not in target_news) and (news not in real_news):
                real_news.append(news)

    num_node = len(target_news) + len(real_news) + len(ctrled_user_list)
    adj_matrix = np.zeros((num_node, num_node))

    for news_id, news in enumerate(target_news):
        users = news_user_mapping[news]
        for user_id, user in enumerate(ctrled_user_list):
            if user in users:
                id1, id2 = news_id, user_id + len(target_news) + len(real_news)
                adj_matrix[id1, id2] += 1.0
                adj_matrix[id2, id1] += 1.0
    
    for news_id, news in enumerate(real_news):
        users = news_user_mapping[news]
        for user_id, user in enumerate(ctrled_user_list):
            if user in users:
                id1, id2 = news_id + len(target_news), user_id + len(target_news) + len(real_news)
                adj_matrix[id1, id2] += 1.0
                adj_matrix[id2, id1] += 1.0
    
    for news_id in range(len(target_news) + len(real_news)):
        if sum(adj_matrix[news_id]) == 0.0:
            for user_id in range(len(target_news) + len(real_news), num_node):
                adj_matrix[news_id, user_id] = 1 / len(ctrled_user_list)
                adj_matrix[user_id, news_id] = 1 / len(ctrled_user_list)
        else:
            adj_matrix[news_id] /= sum(adj_matrix[news_id])
    
    y = PartitionTree(adj_matrix=adj_matrix)
    x = y.build_encoding_tree(height)

    partition_dict = dict()

    for news_id in range(len(target_news)):
        userp, fake_newsp, real_newsp = [], [], []
        pid = y.tree_node[news_id].parent
        news_partition = y.tree_node[pid].partition.copy()
        pid = y.tree_node[pid].parent
        pid = y.tree_node[pid].parent
        user_partition = y.tree_node[pid].partition.copy()
        for nid in news_partition:
            if nid < len(target_news):
                fake_newsp.append(nid)
            elif nid < (len(target_news) + len(real_news)):
                real_newsp.append(nid)
        for nid in user_partition:
            if nid >= (len(target_news) + len(real_news)):
                userp.append(nid)
        if len(userp) < 75:
            userp = list(range(len(target_news) + len(real_news), num_node))
        if len(fake_newsp) > 10:
            fake_newsp = [news_id]
        if len(real_newsp) > 10:
            real_newsp = []
        partition_dict[news_id] = [userp, fake_newsp, real_newsp]

    agg_ps = dict()
    for news_id in partition_dict.keys():
        agg_p = [0.0] * len(controlled_user)
        ctrled_users = partition_dict[news_id][0]
        for cu in ctrled_users:
            for agent_id in range(len(controlled_user)):
                # print(cu, agent_id, len(ctrled_user_list), len(controlled_user))
                if ctrled_user_list[cu - len(target_news) - len(real_news)] in controlled_user[agent_id]:
                    agg_p[agent_id] += y.path_entropy(cu, 0.0)
                    break
        agg_p /= sum(agg_p)
        agg_ps[target_news[news_id]] = agg_p
    
    prob_dict = dict()
    for news_id in partition_dict.keys():
        probs = [news_prob_list[news_id]]
        fake_newsp = partition_dict[news_id][1]
        for nsp in fake_newsp:
            if nsp == news_id:
                continue
            probs.append(news_prob_list[nsp])
        prob_dict[target_news[news_id]] = probs
    
    new_partition_dict = dict()
    for news_id, v in partition_dict.items():
        old_userp, old_fake_newsp, old_real_newsp = v[0], v[1], v[2]
        userp, fake_newsp, real_newsp = [], [target_news[news_id]], []
        for ou in old_userp:
            userp.append(ctrled_user_list[ou - len(target_news) - len(real_news)])
        for ofn in old_fake_newsp:
            if ofn != news_id:
                fake_newsp.append(target_news[ofn])
        for orn in old_real_newsp:
            real_newsp.append(real_news[orn - len(target_news)])
        new_partition_dict[target_news[news_id]] = [userp, fake_newsp, real_newsp]
    
    for key, value in new_partition_dict.items():
        userp, fake_newsp, real_newsp = value[0], value[1], value[2]
        new_userp = [[], [], []]
        for up in userp:
            for i in range(len(controlled_user)):
                if up in controlled_user[i]:
                    new_userp[i].append(up)
                    break
        new_partition_dict[key] = [new_userp, fake_newsp, real_newsp]
    return new_partition_dict, prob_dict, agg_ps
