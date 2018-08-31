from __future__ import print_function
from __future__ import division
import numpy as np
import pandas as pd
import sys
from scipy.io import arff
import itertools
import copy


def get_data_meta(arff_name, save_file=False):
    data0, meta0 = arff.loadarff(arff_name)
    meta = {}
    ordered_meta = {}
    for i in range(len(meta0.names())):
        if meta0.types()[i] != 'numeric':
            meta[meta0.names()[i]] = meta0.__getitem__(meta0.names()[i])
        else:
            meta[meta0.names()[i]] = meta0.__getitem__(meta0.names()[i])[:1] + (('<=', '>'), )
        ordered_meta[meta0.names()[i]] = i
    data = pd.DataFrame(data0)
    for i in range(data.shape[1]):
        if meta[data.ix[:, i].name][0] != 'numeric':
            for j in range(data.shape[0]):
                data.ix[j, i] = data.ix[j, i].decode('utf-8')
    if save_file:
        data.to_csv('data_' + arff_name[:-5] + '.csv')
        np.save('meta_' + arff_name[:-5], meta, fix_imports=True)
    return data, meta, ordered_meta


def get_bayes_graph(meta_train, ordered_meta_train):
    bayes_graph = {}
    ordered_node = list(meta_train.keys())
    ordered_node.sort(key=lambda x: ordered_meta_train[x])
    for i in list(meta_train.keys()):
        if i == 'class':
            bayes_graph[i] = [k for k in ordered_node if k != 'class']
        else:
            bayes_graph[i] = []
    for i in bayes_graph['class']:
        print(i, 'class')
    return bayes_graph


def get_parents(node, graph):
    parents = []
    for i in list(graph.keys()):
        if node in graph[i]:
            parents.append(i)
    return parents


def get_empirical_prob(nodes, values, data_train, meta_train, target_num=1, laplace=1):
    # The first "target_num" entries of "nodes" are the target nodes and the others are the nodes we condition on.
    subset_index = np.array([1] * data_train.shape[0])
    for i in range(target_num, len(nodes)):
        subset_index &= (data_train.ix[:, nodes[i]] == values[i])
    laplace_count = 1
    for i in range(target_num):
        laplace_count *= len(meta_train[nodes[i]][1])
    denominator = sum(subset_index) + laplace * laplace_count
    for i in range(target_num):
        subset_index &= (data_train.ix[:, nodes[i]] == values[i])
    numerator = sum(subset_index) + laplace
    return numerator / denominator


def learning(graph, data_train, meta_train, laplace=1):
    param = {}
    for node in list(graph.keys()):
        parents = get_parents(node, graph)
        nodes = [node] + list(parents)
        param[tuple(nodes)] = {}

        all_values = []
        for i in parents:
            all_values.append(meta_train[i][1])
        all_values_comb = list(itertools.product(*all_values))

        for i in meta_train[node][1]:
            for j in all_values_comb:
                values = [i] + list(j)
                param[tuple(nodes)][tuple(values)] = get_empirical_prob(nodes, values, data_train, meta_train, 1, laplace)

    return param


def get_joint_prob(nodes_values, param):
    # This function returns the joint probability of all nodes.
    # It cannot return the joint probability of any subset of nodes, which is unnecessary in this homework.
    selected_param = []
    for i in list(param.keys()):
        value = [nodes_values[k] for k in i]
        for j in list(param[i].keys()):
            if j == tuple(value):
                selected_param.append(param[i][j])
    return np.prod(np.array(selected_param))


def get_posterior_y(nodes_values, param, meta):
    # This function returns the posterior of y, i.e., the node named "class".
    # It cannot return the posterior of any node, which is unnecessary in this homework.
    nodes_values_copy = copy.deepcopy(nodes_values)
    numerator = get_joint_prob(nodes_values, param)
    denominator = numerator
    for i in meta['class'][1]:
        if i != nodes_values['class']:
            nodes_values_copy['class'] = i
            denominator += get_joint_prob(nodes_values_copy, param)
    return numerator / denominator


def inference(data_test, meta_test, param):
    correct_num = 0
    for i in range(data_test.shape[0]):
        nodes_values = dict(data_test.ix[i, :])
        true_y = nodes_values['class']
        nodes_values_copy = copy.deepcopy(nodes_values)
        max_posterior_y = -np.inf
        predicted_y = 'None'
        for j in meta_test['class'][1]:
            nodes_values_copy['class'] = j
            current_posterior_y = get_posterior_y(nodes_values_copy, param, meta_test)
            if current_posterior_y > max_posterior_y:
                max_posterior_y = current_posterior_y
                predicted_y = j
        if predicted_y == true_y:
            correct_num += 1
        if predicted_y[0] == "'":
            print(predicted_y[1:-1], true_y[1:-1], '{:.12f}'.format(max_posterior_y))
        else:
            print(predicted_y, true_y, '{:.12g}'.format(max_posterior_y))

    print('\n')
    print(correct_num)
    return correct_num


def get_conditional_mi(xi_xj, data_train, meta_train, laplace=1):
    # xi_xj must be a list of length 2.
    conditional_mi = 0
    for i in meta_train[xi_xj[0]][1]:
        for j in meta_train[xi_xj[1]][1]:
            for k in meta_train['class'][1]:
                p1 = get_empirical_prob(xi_xj + ['class'], [i, j, k], data_train, meta_train, 3, laplace)
                p2 = get_empirical_prob(xi_xj + ['class'], [i, j, k], data_train, meta_train, 2, laplace)
                p3 = get_empirical_prob([xi_xj[0]] + ['class'], [i, k], data_train, meta_train, 1, laplace)
                p4 = get_empirical_prob([xi_xj[1]] + ['class'], [j, k], data_train, meta_train, 1, laplace)
                conditional_mi += p1 * np.log2(p2 / (p3 * p4))
    return conditional_mi


def get_graph_weight(data_train, meta_train, laplace=1):
    graph_weight = {}
    for i in list(meta_train.keys()):
        for j in list(meta_train.keys()):
            if (i != j) and (i != 'class') and (j != 'class') and (j, i) not in list(graph_weight.keys()):
                graph_weight[(i, j)] = get_conditional_mi([i, j], data_train, meta_train, laplace)
    return graph_weight


def get_candidate_edges(all_edges, v_new):
    candidate_edges = []
    for i in all_edges:
        if (i[0] in v_new) and (i[1] not in v_new):
            candidate_edges.append((i[0], i[1]))
        elif (i[1] in v_new) and (i[0] not in v_new):
            candidate_edges.append((i[1], i[0]))
    return candidate_edges


def get_direction(node, graph):
    if len(graph[node]) == 0:
        return
    else:
        for i in graph[node]:
            graph[i].remove(node)
    for i in graph[node]:
        get_direction(i, graph)


def get_tan_graph(data_train, meta_train, ordered_meta_train, laplace=1):
    tan_graph = dict(zip(list(meta_train.keys()), [[] for _ in range(len(meta_train))]))
    ordered_node = list(meta_train.keys())
    ordered_node.sort(key=lambda x: ordered_meta_train[x])

    # Step 1
    graph_weight = get_graph_weight(data_train, meta_train, laplace)

    # Step 2
    all_edges = list(graph_weight.keys())
    v_new = {ordered_node[0]}
    while len(v_new) < (len(meta_train) - 1):
        candidate_edges = get_candidate_edges(all_edges, v_new)
        max_weight = -np.inf
        for i in candidate_edges:
            if i not in all_edges:
                j = (i[1], i[0])
            else:
                j = (i[0], i[1])
            if graph_weight[j] > max_weight:
                max_weight = graph_weight[j]
                added_edges = [i]
            elif graph_weight[j] == max_weight:  # One may use a threshold to compare these two values.
                added_edges.append(i)
        if len(added_edges) > 1:  # There is a tie.
            added_edges.sort(key=lambda x: (ordered_meta_train[x[0]], ordered_meta_train[x[1]]))
        v_new.update({added_edges[0][0], added_edges[0][1]})
        tan_graph[added_edges[0][0]].append(added_edges[0][1])
        tan_graph[added_edges[0][1]].append(added_edges[0][0])

    # Step 3
    get_direction(ordered_node[0], tan_graph)

    # Step 4
    tan_graph['class'] = [i for i in list(meta_train.keys()) if i != 'class']

    for i in ordered_node:
        if i != 'class':
            if len(get_parents(i, tan_graph)) == 1:
                print(i, get_parents(i, tan_graph)[0])
            elif len(get_parents(i, tan_graph)) == 2:
                if get_parents(i, tan_graph)[0] != 'class':
                    print(i, get_parents(i, tan_graph)[0], 'class')
                else:
                    print(i, get_parents(i, tan_graph)[1], 'class')

    return tan_graph
