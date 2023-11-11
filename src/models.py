import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import ModuleList
from torch_geometric.nn import SplineConv, GraphConv, GCNConv
from torch_geometric.nn import global_mean_pool
from torchmetrics.functional import auroc, mean_absolute_error


class GraphConvNet(pl.LightningModule):
    def __init__(self, num_features, hidden_features):
        super().__init__()
        self.conv1 = GraphConv(num_features, hidden_features)
        self.conv2 = GraphConv(hidden_features, hidden_features)
        self.conv3 = GraphConv(hidden_features, hidden_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #if the edge_weight attribute exists use it
        if hasattr(data, 'edge_weight'):
            edge_weight = data.edge_weight
        else:
            edge_weight = None

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        return global_mean_pool(x, batch)  # [batch_size, hidden_channels]


class GCNConvNet(pl.LightningModule):
    def __init__(self, num_features, hidden_features):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, hidden_features)
        self.conv3 = GCNConv(hidden_features, hidden_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #if the edge_weight attribute exists use it
        if hasattr(data, 'edge_weight'):
            edge_weight = data.edge_weight
        else:
            edge_weight = None

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = self.conv3(x, edge_index, edge_weight)
        return global_mean_pool(x, batch)  # [batch_size, hidden_channels]


class SplineConvNet(pl.LightningModule):
    def __init__(self, num_features, hidden_features):
        super().__init__()
        self.conv1 = SplineConv(num_features, hidden_features, dim=3, kernel_size=5, aggr='add')
        self.conv2 = SplineConv(hidden_features, hidden_features, dim=3, kernel_size=5, aggr='add')
        self.conv3 = SplineConv(hidden_features, hidden_features, dim=3, kernel_size=5, aggr='add')

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        #if the edge_attr attribute exists use it
        if hasattr(data, 'edge_attr'):
            edge_attr = data.edge_attr
        else:
            edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return global_mean_pool(x, batch)  # [batch_size, hidden_channels]


class MultiGraphClassificationNet(pl.LightningModule):
    def __init__(self, num_features, hidden_features, num_outputs, num_graphs, sub_model, shared_sub_model):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_graphs = num_graphs
        self.sub_model_type = sub_model
        self.sub_models = ModuleList()
        self.shared_sub_model = shared_sub_model
        num_sub_models = 1 if self.shared_sub_model else num_graphs
        for i in range(0,num_sub_models):
            model = sub_model(num_features=num_features, hidden_features=hidden_features)
            self.sub_models.append(model)
        self.fc1 = Linear(num_graphs * hidden_features, hidden_features)
        self.fc2 = Linear(hidden_features, num_outputs)
        self.embeddings = []
        self.fc1_out = []
        self.fc2_out = []
        self.predictions = []
        self.targets = []
        self.subject_ids = []
        self.dataset_names = []

    def forward(self, x, edge_index, data, graph_index):
        # Aggregate graph embeddings
        embeddings = []
        for i, graph in enumerate(data['x']):
            model_idx = 0 if self.shared_sub_model else i
            if i==graph_index:
                g = graph.clone()
                g.x, g.edge_index = x, edge_index
                e = self.sub_models[model_idx](g)
            else:
                e = self.sub_models[model_idx](graph)
            embeddings.append(e)
        x = torch.hstack(embeddings)

        # Apply final classifier
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

    def forward_batch(self, data):
        # Aggregate graph embeddings
        embeddings = []
        for i, graph in enumerate(data['x']):
            model_idx = 0 if self.shared_sub_model else i
            e = self.sub_models[model_idx](graph)
            embeddings.append(e)
        emb = torch.hstack(embeddings)

        # Apply final classifier
        x1 = self.fc1(emb)
        x = F.relu(x1)
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x), x1, emb

    def forward_edge_mask(self, edge_mask, data, graph_index):
        data['x'][graph_index].edge_weight=edge_mask
        return self.forward_batch(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def process_batch(self, batch):
        out, _, _ = self.forward_batch(batch)
        lab = batch['y'].squeeze()
        loss = F.cross_entropy(out, lab)
        prob = torch.softmax(out, dim=1)
        auc = auroc(prob, lab, num_classes=self.num_outputs, average='macro', task='multiclass')
        return loss, auc

    def training_step(self, batch, batch_idx):
        loss, auc = self.process_batch(batch)
        self.log('train_loss', loss, batch_size=len(batch['y']))
        self.log('train_auc', auc, prog_bar=True, batch_size=len(batch['y']))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, auc = self.process_batch(batch)
        self.log('val_loss', loss, batch_size=len(batch['y']))
        self.log('val_auc', auc, prog_bar=True, batch_size=len(batch['y']))

    def on_test_start(self):
        self.embeddings = []
        self.fc1_out = []
        self.fc2_out = []
        self.predictions = []
        self.targets = []
        self.subject_ids = []
        self.dataset_names = []

    def test_step(self, batch, batch_idx):
        out, fc1, emb = self.forward_batch(batch)
        prob = torch.softmax(out, dim=1)
        self.embeddings.append(emb)
        self.fc1_out.append(fc1)
        self.fc2_out.append(out)
        self.predictions.append(prob)
        self.targets.append(batch['y'].squeeze())
        self.subject_ids.append(batch['id'])
        self.dataset_names.append(batch['dataset'])


class MultiGraphRegressionNet(pl.LightningModule):
    def __init__(self, num_features, hidden_features, num_outputs, num_graphs, sub_model, shared_sub_model):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_graphs = num_graphs
        self.sub_model_type = sub_model
        self.sub_models = ModuleList()
        self.shared_sub_model = shared_sub_model
        num_sub_models = 1 if self.shared_sub_model else num_graphs
        for i in range(0,num_sub_models):
            model = sub_model(num_features=num_features, hidden_features=hidden_features)
            self.sub_models.append(model)
        self.fc1 = Linear(num_graphs * hidden_features, hidden_features)
        self.fc2 = Linear(hidden_features, num_outputs)
        self.embeddings = []
        self.fc1_out = []
        self.fc2_out = []
        self.predictions = []        
        self.targets = []
        self.subject_ids = []
        self.dataset_names = []
    
    def forward(self, x, edge_index, data, graph_index):
        self.subject_ids.append(data['id'])
        # Aggregate graph embeddings
        embeddings = []
        for i, graph in enumerate(data['x']):
            model_idx = 0 if self.shared_sub_model else i
            if i==graph_index:
                g = graph.clone()
                g.x, g.edge_index = x, edge_index
                e = self.sub_models[model_idx](g)
            else:
                e = self.sub_models[model_idx](graph)
            embeddings.append(e)
        x = torch.hstack(embeddings)

        # Apply final classifier
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

    def forward_batch(self, data):
        # Aggregate graph embeddings
        embeddings = []
        for i, graph in enumerate(data['x']):
            model_idx = 0 if self.shared_sub_model else i
            e = self.sub_models[model_idx](graph)
            embeddings.append(e)
        emb = torch.hstack(embeddings)

        # Apply final classifier
        x1 = self.fc1(emb)
        x = F.relu(x1)
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x), x1, emb    

    def forward_edge_mask(self, edge_mask, data, graph_index):
        data['x'][graph_index].edge_weight=edge_mask
        return self.forward_batch(data)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def process_batch(self, batch):
        out, _, _ = self.forward_batch(batch)
        out = out.squeeze()
        lab = batch['y'].squeeze()
        loss = F.mse_loss(out, lab)
        mae = mean_absolute_error(out, lab)
        return loss, mae

    def training_step(self, batch, batch_idx):
        loss, mae = self.process_batch(batch)
        self.log('train_loss', loss, batch_size=len(batch['y']))
        self.log('train_mae', mae, prog_bar=True, batch_size=len(batch['y']))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae = self.process_batch(batch)
        self.log('val_loss', loss, batch_size=len(batch['y']))
        self.log('val_mae', mae, prog_bar=True, batch_size=len(batch['y']))

    def on_test_start(self):
        self.embeddings = []
        self.fc1_out = []
        self.fc2_out = []
        self.predictions = []
        self.targets = []
        self.subject_ids = []
        self.dataset_names = []

    def test_step(self, batch, batch_idx):
        out, fc1, emb = self.forward_batch(batch)
        self.embeddings.append(emb)
        self.fc1_out.append(fc1)
        self.fc2_out.append(out)
        self.predictions.append(out.squeeze())
        self.targets.append(batch['y'].squeeze())
        self.subject_ids.append(batch['id'])
        self.dataset_names.append(batch['dataset'])