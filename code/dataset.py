from sys import prefix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, default_collate, random_split
from tqdm import tqdm
import dgl
import json
from gensim.models import Word2Vec, KeyedVectors


def load_ast_codetext_pairs(train_examples, ast_train_path, src_train_path_list):
    ast_lines = open(ast_train_path, 'r', encoding='utf-8').read().splitlines()[:train_examples]
    src_file_paths = open(src_train_path_list, 'r', encoding='utf-8').read().splitlines()[:train_examples]
    src_prefix = '../src_py150/'

    pairs = []
    for ast_line, src_rel_path in tqdm(zip(ast_lines, src_file_paths), desc=f'loading {train_examples} nums of training_examples'):
        pairs.append((ast_line, open(src_prefix + src_rel_path, 'r').read()))
    return pairs

class NodePredictionDataset(Dataset):
    def __init__(self, 
                 train_examples,
                 ast_train_path, 
                 src_train_path_list, 
                 ast_prefix_length # the number of reserved ast nodes before current node (Pre-Ordered)
                 ) -> None:
        super().__init__()

        self.astnode_w2v = KeyedVectors.load('w2v_astnodes.bin')
        self.codetext_w2v = KeyedVectors.load('w2v_codetext.bin')

        self.codetexts = [] # List of str
        self.partial_asts = [] # List of dgl.graph
        self.labels = [] # List of str(node type)

        pairs = load_ast_codetext_pairs(train_examples, ast_train_path, src_train_path_list)
        for ast_line, codetext in pairs:
            ast_nodes = json.loads(ast_line)
            parAST_nxt_pairs = []

            prefix = []
            self.first_order_traversal(ast_nodes, 0, prefix, parAST_nxt_pairs)

            for par_ast, nxt_node in parAST_nxt_pairs:
                self.codetexts.append(self.intercept_codetext_basedon_partialAST(codetext, par_ast))
                self.partial_asts.append(self.parAST_to_dgl_graph(par_ast, ast_prefix_length))
                self.labels.append(ast_nodes[nxt_node]['type'])


    def parAST_to_dgl_graph(self, ast_nodes, ast_prefix_length):
        # TODO: transition partial ast to dgl.graph
        return ast_nodes

    def first_order_traversal(self, ast_nodes, cur_idx, prefix, result):
        cur_node = ast_nodes[cur_idx]
        result.append((prefix, cur_idx))
        if cur_node.get('children'):
            prefix.append(cur_node)
            for child_idx in cur_node['children']:
                self.first_order_traversal(ast_nodes, child_idx, prefix, result)
            prefix.pop()


    def intercept_codetext_basedon_partialAST(self, codetext, ast_nodes):
        # TODO: Intercept the codetext, to make it corresponding to the partial ast.
        return codetext

    def __len__(self):
        return len(self.codetexts)

    def __getitem__(self, index):
        return super().__getitem__(index)