import torch
import torch.nn as nn

import dgl


class BinaryTreeLSTMCell(nn.Module):
    
    def __init__(self, x_size, h_size):
        super().__init__()
        
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(2 * h_size, 2 * h_size)
    
    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}
    
    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size()[0], -1)
        f = torch.sigmoid(self.U_f(h_cat).view(*nodes.mailbox['h'].size()))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}
    
    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}
    

class ChildSumTreeLSTMCell(nn.Module):
    
    def __init__(self, x_size, h_size) -> None:
        super().__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bais=False)
        self.b_iou = nn.Parameter(torch.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)
    
    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}
    
    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], axis=1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'])
        return {'iou': self.U_iou(h_tild), 'c': c}
    
    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}
    
    
class BinaryTreeLSTM(nn.Module):
    
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size, 
                 dropout,
                 pretrained_emb=None):
        super().__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        
        self.dropout = nn.Dropout(dropout)
        self.cell = BinaryTreeLSTMCell(x_size, h_size)

    def forward(self, graphs: dgl.graph, h: torch.Tensor, c: torch.Tensor):
        """Compute tree-lstm prediction given a batch.

        Parameters
        ----------
        graphs
            The batched graph, use `dgl.unbatch` to unbatch into list of graphs.
        h : Tensor
            Initial hidden state, shape: (num_nodes, h_size).
        c : Tensor
            Initial cell state, shape: (num_nodes, h_size).

        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        
        
        # Caculate mean of subtoken embeddings as node embedding
        # indices: (num_nodes, MAX_SUBTOKEN_NUMBER)
        # mask: (num_nodes, MAX_SUBTOKEN_NUMBER)
        # ==> 
        # subtoken_embeds: (num_nodes, MAX_SUBTOKEN_NUMBER, embedding_size)
        indices = graphs.ndata['subtokens']
        mask = graphs.ndata['mask']
        subtoken_embeds = self.embedding(indices) * torch.unsqueeze(mask, dim=-1)
        mean_embeds = torch.sum(subtoken_embeds, dim=1) / torch.sum(mask, dim=1, keepdim=True)
        mean_embeds = torch.nan_to_num(mean_embeds, nan=0.)
        
        node_embeds = self.dropout(mean_embeds)
        graphs.ndata['iou'] = self.cell.W_iou(node_embeds)
        graphs.ndata['h'] = h
        graphs.ndata['c'] = c
        
        dgl.prop_nodes_topo(graphs,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        
        # Extract all root node embeddings
        graphs = dgl.unbatch(graphs)
        root_embeddings = [g.ndata['h'][0] for g in graphs]
        h = torch.stack(root_embeddings)
        return h
        
                
