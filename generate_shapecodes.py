import argparse
import numpy as np
import os
import os.path as osp
import torch
import sys
import importlib
import open3d as o3d
from tqdm import tqdm

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

##UMUR##
def readData(data_path):
    mesh_paths = []
    # giving file extension
    ext = ('.obj')
    for file_name in os.listdir(data_path):
        if file_name.endswith(ext):
            file_path = osp.join(data_path, file_name)
            mesh_paths.append(file_path)
            print(f"{file_path} added to mesh_paths list")  # printing file name of desired extension
        else:
            print("Extension not found in the possible matches:" + ext)
    return mesh_paths


def loadModel(args, model_name, checkpoint_path):
    num_class = args.num_category
    model = importlib.import_module(model_name)
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    print(f'{model_name} is loaded')
    if not args.use_cpu:
        classifier = classifier.cuda()
    checkpoint = torch.load(checkpoint_path)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    return classifier
    
    
def getPointCloudFromMeshPath(args, mesh_path, noOfSamplingPoints):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    pcd = mesh.sample_points_uniformly(number_of_points=noOfSamplingPoints)
    if(args.use_normals):
        pcd.estimate_normals()
    points = torch.tensor(pcd.points).float()
    points = preprocess_points(points)
    print(f'Points shape: {points.shape}')
    return points


def generateEncodingAndSave(classifier, points, mesh_path):
    with torch.no_grad():
        _,_, encoding = classifier(points)
        # print(f'Encoding: {encoding}')
        print(f'Encoding shape: {encoding.shape}')

        fileNameToWrite = os.path.splitext(mesh_path)[0] + '.npy'
        with open(fileNameToWrite, 'wb') as f:
            np.save(f, encoding.cpu())
            print("Saved as " + fileNameToWrite)
    
    
##END_UMUR##


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def preprocess_points(points):
    points_min, points_max = points.min(), points.max()
    points = (points - points_min)/(points_max - points_min)
    points = points[None, :]
    points = points.transpose(2, 1)
    return points


def parse_args():
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--log_dir', type=str, default='pointnet2_ssg_wo_normals', help='Experiment root')
    parser.add_argument('--data_path', type=str, required=True, default='data/adl_shapenet', help='give data path')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args = parse_args()
    checkpoint_path = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    noOfSamplingPoints = 1024
    model_name = 'pointnet2_cls_ssg'
    
    ##UMUR##
    #parse data path argument
    data_path = args.data_path
    for label in tqdm(os.listdir(data_path)):
        label = 'monitor'
        class_path = osp.join(data_path, label)
        mesh_paths = readData(class_path)
    
        '''MODEL LOADING'''
        classifier = loadModel(args, model_name, checkpoint_path)
        
        for mesh_path in mesh_paths:
            '''INPUT'''
            points = getPointCloudFromMeshPath(args, mesh_path, noOfSamplingPoints)
            if not args.use_cpu:
                points = points.cuda()
            '''GENERATE ENCODINGS'''
            generateEncodingAndSave(classifier, points, mesh_path)
        exit(0)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)


