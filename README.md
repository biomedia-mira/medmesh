# A Comparative Study of Graph Neural Networks for Shape Classification in Neuroimaging

This repository contains the code for the paper
> N. Shehata, W. Bain, B. Glocker. [_A Comparative Study of Graph Neural Networks for Shape Classification in Neuroimaging_](https://openreview.net/forum?id=HdCrxrSXZZ-). Proceedings of Machine Learning Research. GeoMedIA Workshop. 2022

## Datasets

The brain imaging datasets used in the paper are all publicly available but cannot be shared directly by us. Please see the following websites for accessing the original data.

- UK Biobank data: https://www.ukbiobank.ac.uk/
- CAM-CAN data: https://www.cam-can.org/
- IXI dataset: https://brain-development.org/ixi-dataset/
- OASIS3 dataset: https://www.oasis-brains.org/

UK Biobank provides pre-processed data including the meshes of the subcortical structures. All other datasets were processed by us using [FSL-FIRST](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide) to segment the 15 different brain structures.

We can only provide the meshes for the IXI dataset, allowing for the code to be run for an example of sex classification when trained and tested on IXI subsets. Please [download the IXI meshes](https://) and unzip in the `data/brain` subfolder.

## Code

For running the code, we recommend setting up a dedicated Python environment.

### Setup Python environment using conda

Create and activate a Python 3 conda environment:

   ```shell
   conda create -n pymesh python=3.8
   conda activate pymesh
   ```
   
Install PyTorch using conda (for CUDA Toolkit 11.3):
   
   ```shell
   conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
   ```

Install PyTorch Geometric:
   
   ```shell
   conda install pyg -c pyg -c conda-forge
   ```

### Setup Python environment using virtualenv

Create and activate a Python 3 virtual environment:

   ```shell
   virtualenv -p python3.8 <path_to_envs>/pymesh
   source <path_to_envs>/pymesh/bin/activate
   ```
   
Install PyTorch using pip:
   
   ```shell
   pip install torch torchvision
   ```

Install PyTorch Geometric:
   
   ```shell
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
   ```

### Install additional Python packages:
   
   ```shell
   pip install matplotlib jupyter pandas seaborn scikit-learn tensorboard pytorch-lightning cmake openmesh vtk pyvista open3d
   ```

### How to use

In order to train and test an example sex classification model with the IXI dataset:

1. Download the [IXI meshes](https://), unzip in the `data/brain` subfolder.
2. Run the the script [`brain_shape_classification.py`](brain_shape_classification.py).

## License
This project is licensed under the [Apache License 2.0](LICENSE).
