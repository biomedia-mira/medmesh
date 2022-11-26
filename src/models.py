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
        x, edge_index, pseudo, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = F.relu(self.conv1(x, edge_index, pseudo))
        x = F.relu(self.conv2(x, edge_index, pseudo))
        x = self.conv3(x, edge_index, pseudo)
        return global_mean_pool(x, batch)  # [batch_size, hidden_channels]


class MultiGraphClassificationNet(pl.LightningModule):
    def __init__(self, num_features, hidden_features, num_outputs, num_graphs, sub_model, shared_sub_model):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_graphs = num_graphs
        self.sub_models = ModuleList()
        self.shared_sub_model = shared_sub_model
        num_sub_models = 1 if self.shared_sub_model else num_graphs
        for i in range(0,num_sub_models):
            model = sub_model(num_features=num_features, hidden_features=hidden_features)
            self.sub_models.append(model)
        self.fc1 = Linear(num_graphs * hidden_features, hidden_features)
        self.fc2 = Linear(hidden_features, num_outputs)
        self.predictions = []
        self.targets = []

    def forward(self, data):
        # Aggregate graph embeddings
        embeddings = []
        for i, graph in enumerate(data['x']):
            model_idx = 0 if self.shared_sub_model else i
            x = self.sub_models[model_idx](graph)
            embeddings.append(x)
        x = torch.hstack(embeddings)

        # Apply final classifier
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def process_batch(self, batch):
        out = self.forward(batch)
        lab = batch['y'].squeeze()
        loss = F.cross_entropy(out, lab)
        prob = torch.softmax(out, dim=1)
        auc = auroc(prob, lab, num_classes=self.num_outputs, average='macro')
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
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        prob = torch.softmax(out, dim=1)
        self.predictions.append(prob)
        self.targets.append(batch['y'].squeeze())


class MultiGraphRegressionNet(pl.LightningModule):
    def __init__(self, num_features, hidden_features, num_outputs, num_graphs, sub_model, shared_sub_model):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_graphs = num_graphs
        self.sub_models = ModuleList()
        self.shared_sub_model = shared_sub_model
        num_sub_models = 1 if self.shared_sub_model else num_graphs
        for i in range(0,num_sub_models):
            model = sub_model(num_features=num_features, hidden_features=hidden_features)
            self.sub_models.append(model)
        self.fc1 = Linear(num_graphs * hidden_features, hidden_features)
        self.fc2 = Linear(hidden_features, num_outputs)
        self.predictions = []
        self.targets = []

    def forward(self, data):
        # Aggregate graph embeddings
        embeddings = []
        for i, graph in enumerate(data['x']):
            model_idx = 0 if self.shared_sub_model else i
            x = self.sub_models[model_idx](graph)
            embeddings.append(x)
        x = torch.hstack(embeddings)

        # Apply final classifier
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        return self.fc2(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

    def process_batch(self, batch):
        out = self.forward(batch).squeeze()
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
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        self.predictions.append(out.squeeze())
        self.targets.append(batch['y'].squeeze())

