import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

class Model(torch.nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.nhid = args.nhid
        self.num_classes = args.num_classes
        self.model = args.base_model
        self.num_layers = args.nhop

        if self.model == 'gcn':
            if self.num_layers == 1:
                self.conv1 = GCNConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.conv1 = GCNConv(self.num_features, self.nhid * 2)
                self.conv2 = GCNConv(self.nhid * 2, self.nhid)
        elif self.model == 'sage':
            if self.num_layers == 1:
                self.conv1 = SAGEConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.conv1 = SAGEConv(self.num_features, self.nhid * 2)
                self.conv2 = SAGEConv(self.nhid * 2, self.nhid)
        elif self.model == 'gat':
            if self.num_layers == 1:
                self.conv1 = GATConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.conv1 = GATConv(self.num_features, self.nhid * 2)
                self.conv2 = GATConv(self.nhid * 2, self.nhid)
        elif self.model == 'bi-gcn':
            self.conv_td1 = GCNConv(self.num_features, self.nhid)
            self.conv_bu1 = GCNConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.conv_td2 = GCNConv(self.nhid, self.nhid)
                self.conv_bu2 = GCNConv(self.nhid, self.nhid)
            self.dropout_rate = args.dropout_rate if hasattr(args, "dropout_rate") else 0.2
        elif self.model == 'gcan':
            self.gat1 = GATConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.gat2 = GATConv(self.nhid, self.nhid)
            self.co_att = torch.nn.Linear(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if self.model in ['gcn', 'sage', 'gat']:
            edge_attr = None
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            if self.num_layers == 2:
                x = F.relu(self.conv2(x, edge_index, edge_attr))

        elif self.model == 'bi-gcn':
            edge_index_td = data.edge_index_td
            edge_index_bu = data.edge_index_bu
            x_td = F.relu(self.conv_td1(x, edge_index_td))
            x_bu = F.relu(self.conv_bu1(x, edge_index_bu))

            if self.num_layers == 2:
                x_td = F.relu(self.conv_td2(x_td, edge_index_td))
                x_bu = F.relu(self.conv_bu2(x_bu, edge_index_bu))

            if self.training:
                mask = torch.rand_like(x_td) > self.dropout_rate
                x_td = x_td * mask
                x_bu = x_bu * mask

            x_combined = x_td + x_bu

            x = x_combined + x

        elif self.model == 'gcan':
            x = F.relu(self.gat1(x, edge_index))
            if self.num_layers == 2:
                x = F.relu(self.gat2(x, edge_index))
            x = F.relu(self.co_att(x))

        x = F.log_softmax(self.lin1(x), dim=-1)
        return x
