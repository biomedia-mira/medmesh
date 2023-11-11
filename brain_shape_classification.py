import os
import torch
import pandas as pd
import torch_geometric.transforms as T
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from torchmetrics.functional import accuracy, auroc

from src.models import MultiGraphClassificationNet, SplineConvNet, GraphConvNet, GCNConvNet
from src.datasets import MeshDataModule, UKBBDataset, CamCANDataset, IXIDataset, OASIS3Dataset
from src.transforms import PositionFeatures, FPFHFeatures, SHOTFeatures


# UK Biobank data https://www.ukbiobank.ac.uk/
# data_dir_ukbb = './data/brain/ukbb/'
# mesh_path_ukbb = data_dir_ukbb + 't0'
# metadata_file_ukbb = data_dir_ukbb + 'ukb21079_extracted.csv'

# CAM-CAN data https://www.cam-can.org/
# data_dir_camcan = './data/brain/camcan/'
# mesh_path_camcan = data_dir_camcan + 'meshes'
# metadata_file_camcan = data_dir_camcan + 'clean_participant_data.csv'

# IXI dataset https://brain-development.org/ixi-dataset/
data_dir_ixi = './data/brain/ixi/'
mesh_path_ixi = data_dir_ixi + 'meshes'
metadata_file_ixi = data_dir_ixi + 'IXI.csv'

# OASIS3 dataset https://www.oasis-brains.org/
# data_dir_oasis3 = './data/brain/oasis3/'
# mesh_path_oasis3 = data_dir_oasis3 + 'meshes'
# metadata_file_oasis3 = data_dir_oasis3 + 'oasis3_one_scan_per_subject.csv'

cache_path = './data/brain'

train_split = 0.8
val_split = 0.1
substructures = ['BrStem', 'L_Hipp', 'R_Hipp', 'L_Accu', 'R_Accu', 'L_Amyg', 'R_Amyg', 'L_Puta', 'R_Puta', 'L_Thal', 'R_Thal', 'L_Pall', 'R_Pall', 'L_Caud', 'R_Caud']
shared_sub_model = True
sub_model = GCNConvNet

pre_processing = T.Compose([T.FaceToEdge(remove_faces = False)])
augmentation = T.Compose([T.RandomJitter(0.1)])
features = T.Compose([T.Constant(value=1), T.Spherical(), FPFHFeatures()])

train_transform = T.Compose([pre_processing, augmentation, features])
test_transform = T.Compose([pre_processing, features])

target_label = 'sex'

epochs = 30
batch_size = 128
num_workers = 4
hidden_features = 32
num_classes = 2


def save_predictions(model, output_fname):

    embds = torch.cat(model.embeddings, dim=0)
    fc1 = torch.cat(model.fc1_out, dim=0)
    fc2 = torch.cat(model.fc2_out, dim=0)
    preds = torch.cat(model.predictions, dim=0)
    targets = torch.cat(model.targets, dim=0)
    subject_ids = [id for sublist in model.subject_ids for id in sublist]
    dataset_names = [dataset for sublist in model.dataset_names for dataset in sublist]

    counts = []
    for i in range(0, num_classes):
        t = targets == i
        c = torch.sum(t)
        counts.append(c)
    print(counts)

    auc_per_class = auroc(preds, targets, num_classes=num_classes, average='none', task='multiclass')
    auc_avg = auroc(preds, targets, num_classes=num_classes, average='macro', task='multiclass')
    acc_per_class = accuracy(preds, targets, num_classes=num_classes, average='none', task='multiclass')
    acc_avg = accuracy(preds, targets, num_classes=num_classes, average='macro', task='multiclass')

    print('AUC per-class')
    print(auc_per_class)

    print('AUC average')
    print(auc_avg)

    print('ACC per-class')
    print(acc_per_class)

    print('ACC average')
    print(acc_avg)

    cols_names_preds = ['class_' + str(i) for i in range(0, num_classes)]
    cols_names_fc1 = ['fc1_' + str(i) for i in range(0, fc1.size(dim=1))]
    cols_names_fc2 = ['fc2_' + str(i) for i in range(0, fc2.size(dim=1))]
    cols_names_embds = ['embds_' + str(i) for i in range(0, embds.size(dim=1))]
    
    df = pd.DataFrame(data=preds.cpu().numpy(), columns=cols_names_preds)    
    df['target'] = targets.cpu().numpy()
    df['subject_id'] = subject_ids
    df['dataset_name'] = dataset_names
    df_embds = pd.DataFrame(data=embds.cpu().numpy(), columns=cols_names_embds)
    df_fc1 = pd.DataFrame(data=fc1.cpu().numpy(), columns=cols_names_fc1)
    df_fc2 = pd.DataFrame(data=fc2.cpu().numpy(), columns=cols_names_fc2)
    df = pd.concat([df, df_embds, df_fc1, df_fc2], axis=1)
    df.to_csv(output_fname, index=False)


def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    # Train on IXI dataset
    data = MeshDataModule(dataset=IXIDataset, data_path=mesh_path_ixi, metadata_file=metadata_file_ixi, target_label=target_label,
                          cache_path=cache_path, train_split=train_split, val_split=val_split,
                          substructures=substructures, train_transform=train_transform, test_transform=test_transform,
                          batch_size=batch_size, num_workers=num_workers)

    # Setup from the paper, train on UK Biobank
    # data = MeshDataModule(dataset=UKBBDataset, data_path=mesh_path_ukbb, metadata_file=metadata_file_ukbb, target_label=target_label,
    #                       cache_path=cache_path, train_split=train_split, val_split=val_split,
    #                       substructures=substructures, train_transform=train_transform, test_transform=test_transform,
    #                       batch_size=batch_size, num_workers=num_workers)

    print()
    print('=============================================================')
    print(f'Number of samples: {len(data.dataset)}')
    print(f'Training: {len(data.train_set)}')
    print(f'Validation: {len(data.val_set)}')
    print(f'Testing: {len(data.test_set)}')
    print()
    print('=============================================================')
    sample = data.dataset[0]['x']
    num_features = sample[0].num_features
    print(f'Number of sub-graphs: {len(sample)}')
    print(f'Number of node features: {num_features}')
    for i, graph in enumerate(sample):
        print('-------------------------------------------------------------')
        print(graph)
        print(f'Structure: {i} - {substructures[i]}')
        print(f'Number of nodes: {graph.num_nodes}')
        print(f'Number of edges: {graph.num_edges}')
        print(f'Average node degree: {graph.num_edges / graph.num_nodes:.2f}')
        print(f'Contains isolated nodes: {graph.has_isolated_nodes()}')
        print(f'Contains self-loops: {graph.has_self_loops()}')
        print(f'Is undirected: {graph.is_undirected()}')
    print('=============================================================')

    # Create output directory
    out_name = sub_model.__name__ + '-FPFH-shared-no-reg'
    out_dir = 'output/' + target_label + '/' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # model
    model = MultiGraphClassificationNet(num_features=num_features, hidden_features=hidden_features, num_outputs=num_classes, num_graphs=len(substructures), sub_model=sub_model, shared_sub_model=shared_sub_model)

    # train
    checkpoint_callback = ModelCheckpoint(monitor="val_auc", mode='max')
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps=5,
        max_epochs=epochs,
        accelerator=hparams.dev,
        devices=1,
        logger=TensorBoardLogger('output/' + target_label, name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = MultiGraphClassificationNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_features=num_features, hidden_features=hidden_features, num_outputs=num_classes, num_graphs=len(substructures), sub_model=sub_model, shared_sub_model=shared_sub_model)

    print('=============================================================')
    print('Testing on IXI')

    trainer.test(model=model, datamodule=data)
    save_predictions(model=model, output_fname=os.path.join(out_dir, 'predictions_test_ixi.csv'))
    

    # Setup from the paper, train on UK Biobank and test on all

    # print('=============================================================')
    # print('Testing on UKBB')

    # trainer.test(model=model, datamodule=data)
    # save_predictions(model=model, output_fname=os.path.join(out_dir, 'predictions_test_ukbb.csv'))


    # print('=============================================================')
    # print('Testing on CamCAN')

    # data = MeshDataModule(dataset=CamCANDataset, data_path=mesh_path_camcan, metadata_file=metadata_file_camcan, target_label=target_label,
    #                       cache_path=cache_path, train_split=0, val_split=0,
    #                       substructures=substructures, train_transform=train_transform, test_transform=test_transform,
    #                       batch_size=batch_size, num_workers=num_workers)

    # trainer.test(model=model, datamodule=data)
    # save_predictions(model=model, output_fname=os.path.join(out_dir, 'predictions_test_camcan.csv'))


    # print('=============================================================')
    # print('Testing on IXI')

    # data = MeshDataModule(dataset=IXIDataset, data_path=mesh_path_ixi, metadata_file=metadata_file_ixi, target_label=target_label,
    #                       cache_path=cache_path, train_split=0, val_split=0,
    #                       substructures=substructures, train_transform=train_transform, test_transform=test_transform,
    #                       batch_size=batch_size, num_workers=num_workers)

    # trainer.test(model=model, datamodule=data)
    # save_predictions(model=model, output_fname=os.path.join(out_dir, 'predictions_test_ixi.csv'))


    # print('=============================================================')
    # print('Testing on OASIS3')

    # data = MeshDataModule(dataset=OASIS3Dataset, data_path=mesh_path_oasis3, metadata_file=metadata_file_oasis3, target_label=target_label,
    #                       cache_path=cache_path, train_split=0, val_split=0,
    #                       substructures=substructures, train_transform=train_transform, test_transform=test_transform,
    #                       batch_size=batch_size, num_workers=num_workers)

    # trainer.test(model=model, datamodule=data)
    # save_predictions(model=model, output_fname=os.path.join(out_dir, 'predictions_test_oasis3.csv'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dev', default='gpu')
    args = parser.parse_args()

    main(args)