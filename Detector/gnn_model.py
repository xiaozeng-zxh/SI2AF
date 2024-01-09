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
        # Bi-GCN 层的初始化
        elif self.model == 'bi-gcn':
            self.conv1_news = GCNConv(self.num_features, self.nhid)
            self.conv1_user = GCNConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.conv2_news = GCNConv(self.nhid, self.nhid)
                self.conv2_user = GCNConv(self.nhid, self.nhid)
        # GCAN 层的初始化
        elif self.model == 'gcan':
            self.gat1 = GATConv(self.num_features, self.nhid)
            if self.num_layers == 2:
                self.gat2 = GATConv(self.nhid, self.nhid)
            # 简单的协同注意力机制，可以根据需要进行更复杂的设计
            self.co_att = torch.nn.Linear(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.num_classes)

    def forward(self, data):

        x, edge_index = data.x, data.edge_index

        if self.model == 'gcn' or self.model == 'sage' or self.model == 'gat':

            edge_attr = None

            x = F.relu(self.conv1(x, edge_index, edge_attr))
            if self.num_layers == 2:
                x = F.relu(self.conv2(x, edge_index, edge_attr))
        
        elif self.model == 'bi-gcn':
            # Bi-GCN 前向传播逻辑
            x_news = F.relu(self.conv1_news(x, edge_index))
            x_user = F.relu(self.conv1_user(x, edge_index))
            if self.num_layers == 2:
                x_news = F.relu(self.conv2_news(x_news, edge_index))
                x_user = F.relu(self.conv2_user(x_user, edge_index))
            # 合并新闻和用户的特征，此处只是一个简单的例子，实际应用中可能需要不同的合并策略
            x = x_news + x_user

        elif self.model == 'gcan':
            # GCAN 前向传播逻辑
            x = F.relu(self.gat1(x, edge_index))
            if self.num_layers == 2:
                x = F.relu(self.gat2(x, edge_index))
            x = F.relu(self.co_att(x))  # 协同注意力机制

        x = F.log_softmax(self.lin1(x), dim=-1)

        return x
