import torch
import numpy as np
import open3d as o3d
import os
import pyvista as pv
import time


class MultiGraphTransform():
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data):
        if self.transform:
            num_graphs = len(data['x'])
            for i in range(0,num_graphs):
                data['x'][i] = self.transform(data['x'][i])
        return data


class PositionFeatures():
    def __call__(self, data):
        pos_features = data.pos.detach().clone()
        if data.x is None:
            data.x = pos_features.float()
        else:
            data.x = torch.cat((data.x, pos_features), dim = 1).float()
        return data


class FPFHFeatures():
    def __call__(self, data, voxel_size=2):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data.pos)

        radius_normal = voxel_size * 2
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd,
                                                                   o3d.geometry.KDTreeSearchParamHybrid(
                                                                       radius=radius_feature, max_nn=100))

        fpfh_features = torch.tensor(pcd_fpfh.data.transpose(), dtype=torch.float)

        if data.x is None:
            data.x = fpfh_features
        else:
            data.x = torch.cat((data.x, fpfh_features), dim = 1).float()
        return data


class SHOTFeatures():
    def __call__(self, data):
        ply_dir = "./temp/PLY"
        shot_dir = "./temp/SHOT"

        if not os.path.exists(ply_dir):
            os.mkdir(ply_dir)

        if not os.path.exists(shot_dir):
            os.mkdir(shot_dir)

        # plypath is path to ply file
        fname = os.path.basename(data.vtkpath)[:-4]
        plypath = os.path.join(ply_dir, fname + ".ply")
        savepath = os.path.join(shot_dir, fname + "_SHOT.txt")

        pvpolydata = pv.read(data['vtkpath'])

        if not os.path.exists(savepath):
            if not os.path.exists(plypath):
                pvpolydata.save(plypath)

            while not os.path.exists(plypath):
                time.sleep(1)

            # calculate SHOT
            cmd = './ext/SHOT -f -i ' + plypath + ' -r 4 -k ' + str(len(pvpolydata.points)) + ' -o ' + savepath
            os.system(cmd)

        while not os.path.exists(savepath):
            time.sleep(1)

        SHOT = np.loadtxt(savepath)
        SHOT = SHOT[SHOT[:, 0].argsort()]
        SHOT = np.delete(SHOT, [0, 1, 2, 3], 1)
        shot_features = torch.tensor(SHOT, dtype=torch.float)

        if data.x is None:
            data.x = shot_features
        else:
            data.x = torch.cat((data.x, shot_features), dim = 1).float()
        return data
    
class DisplacementFeatures():
    def __call__(self, data):
        ref_pvpolydata = pv.read(data.vtkpath)
        npts = len(ref_pvpolydata.points)
        fname = data.vtkpath[:-6] #remove last two digits and extension (.vtk)
        t_dist = np.zeros(npts)
        for t in range(1,50):
            t_str = str(t)
            t_digits = t_str.zfill(2) #leading zeros for timepoints 1-49
            tmp_fname = fname + t_digits + '.vtk'
            pvpolydata = pv.read(tmp_fname)
            
            p_dists=[]
            for p in range(npts):
                p_dists = np.append(p_dists, np.linalg.norm(pvpolydata.points[p] - ref_pvpolydata.points[p]))

            t_dist = np.vstack( (t_dist, p_dists) )
            
        if data.x is None:
            data.x = torch.tensor(t_dist.transpose(), dtype=torch.float)
        else:
            data.x = torch.cat((data.x, torch.tensor(t_dist.transpose(), dtype=torch.float)), dim = 1).float()
        return data
            
            