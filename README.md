# Graph Neural Networks for Medical Shape Classification

This repository contains the code for the papers
> N. Shehata, W. Bain, B. Glocker. [_A Comparative Study of Graph Neural Networks for Shape Classification in Neuroimaging_](https://openreview.net/forum?id=HdCrxrSXZZ-). Proceedings of Machine Learning Research. GeoMedIA Workshop. 2022

> N. Shehata, C. Picarra, A. Kazi, B. Glocker. _The Importance of Model Inspection for Better Understanding Performance Characteristics of Graph Neural Networks_. Under review. 2023

## Datasets

The brain imaging datasets used in the paper are all publicly available but cannot be shared directly by us. Please see the following websites for accessing the original data.

- UK Biobank data: https://www.ukbiobank.ac.uk/
- CAM-CAN data: https://www.cam-can.org/
- IXI dataset: https://brain-development.org/ixi-dataset/
- OASIS3 dataset: https://www.oasis-brains.org/

UK Biobank provides pre-processed data including the meshes of the subcortical structures. All other datasets were processed by us using [FSL-FIRST](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide) to segment the 15 different brain structures.

We can only provide the meshes for the IXI dataset, allowing for the code to be run for an example of sex classification when trained and tested on IXI subsets. Please [download the IXI meshes](https://imperialcollegelondon.box.com/s/qasj750gwk0e62mncy9pycp2bznsho7m) and unzip in the `data/brain` subfolder.

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3 conda environment:

   ```shell
   conda create -n pymesh python=3.8
   conda activate pymesh
   ```

### Setup Python environment using virtualenv

Create and activate a Python 3 virtual environment:

   ```shell
   virtualenv -p python3.8 <path_to_envs>/pymesh
   source <path_to_envs>/pymesh/bin/activate
   ```

### Install PyTorch

Check out instructions for [how to install PyTorch](https://pytorch.org/get-started/locally/) using conda or pip.

### Install PyTorch Geometric

Check out instructions for [how to install PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#) using conda or pip. Make sure to also install the package `torch-spline-conv`.

### Install additional packages:
   
   ```shell
   pip install matplotlib jupyter pandas seaborn scikit-learn tensorboard pytorch-lightning cmake vtk pyvista open3d
   pip install openmesh
   ```
   
   For conda users, openmesh may have to be installed via
   ```shell
   conda install -c conda-forge openmesh-python
   ```

### How to use

In order to train and test an example sex classification model with the IXI dataset:

1. Download the [IXI meshes](https://imperialcollegelondon.box.com/s/qasj750gwk0e62mncy9pycp2bznsho7m), unzip in the `data/brain` subfolder.
2. Run the the script [`brain_shape_classification.py`](brain_shape_classification.py).

## License
This project is licensed under the [Apache License 2.0](LICENSE).
