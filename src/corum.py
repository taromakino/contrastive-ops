import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
from argparse import ArgumentParser
from scipy.spatial import distance
from sklearn.metrics import recall_score, precision_score


def get_corum_graph(corum_fpath, genes):
    rows = open(corum_fpath).readlines()
    gene_groups = [row.split('\t\t')[1].split('\t') for row in rows]
    [gene_group.remove('\n') for gene_group in gene_groups]
    gene_groups = [[gene for gene in gene_group if gene in genes] for gene_group in gene_groups]
    unique_genes = set(list(itertools.chain(*gene_groups)))
    gene_graph = nx.Graph()
    parent_to_children = {x: set() for x in unique_genes}
    for gene_group in gene_groups:
        for gene in gene_group:
            parent_to_children[gene].update(gene_group)
            parent_to_children[gene].remove(gene)
    for parent, children in parent_to_children.items():
        for child in children:
            gene_graph.add_edge(parent, child)
    return gene_graph


def get_precision_and_recall(embeddings, gene_graph, percentile_range):
    sim_mat = 1 - distance.cdist(embeddings, embeddings, 'cosine')
    sim_mat_flat = sim_mat.flatten()
    target, pred_sim = get_target_and_pred_sim(gene_graph, sim_mat)
    df = {'precision': [], 'recall': []}
    for percentile in percentile_range:
        upper_threshold = np.percentile(sim_mat_flat, percentile)
        lower_threshold = np.percentile(sim_mat_flat, 100 - percentile)
        pred_binary = pred_sim_to_binary(pred_sim, upper_threshold, lower_threshold)
        df['precision'].append(precision_score(target, pred_binary))
        df['recall'].append(recall_score(target, pred_binary))
    return pd.DataFrame(df)


def get_target_and_pred_sim(gene_graph, sim_mat):
    target, pred_sim = [], []
    adj_mat = nx.adjacency_matrix(gene_graph).toarray()
    for i in range(len(adj_mat)):
        target_elem = adj_mat[i].tolist()
        pred_sim_elem = sim_mat[i].tolist()
        target_elem.pop(i)
        pred_sim_elem.pop(i)
        target += target_elem
        pred_sim += pred_sim_elem
    return target, pred_sim


def pred_sim_to_binary(pred, upper_threshold, lower_threshold):
    return [1 if ((elem >= upper_threshold) or (elem <= lower_threshold)) else 0 for elem in pred]


def main(args):
    df = pd.read_pickle(args.embed_fpath)

    df = df.groupby('sgRNA_0').agg(
        {col: 'mean' if col != 'gene_symbol_0' else 'first' for col in df.columns[1:]}
    ).groupby('gene_symbol_0').mean()
    df = df - df.loc['nontargeting']
    df = (df - df.mean(axis=0)) / df.std(axis=0)

    genes = df.index.values
    gene_graph = get_corum_graph(args.corum_fpath, genes)

    idxs = []
    for node in list(gene_graph.nodes):
        idxs.append(np.where(genes == node)[0][0])

    df = df.iloc[idxs]
    assert list(df.index.values) == list(gene_graph.nodes)

    precision_and_recall = get_precision_and_recall(
        df.values,
        gene_graph,
        np.arange(80, 100, 1)
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(precision_and_recall.recall, precision_and_recall.precision, 'o-')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    fig.tight_layout()
    os.makedirs(args.results_dpath, exist_ok=True)
    plt.savefig(os.path.join(args.results_dpath, 'corum.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--results_dpath', type=str, required=True)
    parser.add_argument('--corum_fpath', type=str, required=True)
    parser.add_argument('--embed_fpath', type=str, required=True)
    main(parser.parse_args())