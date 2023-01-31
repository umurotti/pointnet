import argparse
import numpy as np
import os
import torch
import sys
import importlib
import open3d as o3d

##UMUR##
def readData(dataPath):
    meshPaths = []
    # giving file extension
    ext = ('.obj')
    for files in os.listdir(dataPath):
        if files.endswith(ext):
            meshPaths.append(dataPath + files)
            print(dataPath + files + " added to meshPaths list")  # printing file name of desired extension
        else:
            print("Extension not found in the possible matches:" + ext)
    return meshPaths


def loadModel(args, modelName, checkpointPath):
    num_class = args.num_category
    model = importlib.import_module(modelName)
    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    print(f'{modelName} is loaded')
    if not args.use_cpu:
        classifier = classifier.cuda()
    checkpoint = torch.load(checkpointPath)
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()
    return classifier
    
    
def getPointCloudFromMeshPath(args, meshPath, noOfSamplingPoints):
    mesh = o3d.io.read_triangle_mesh(meshPath)
    pcd = mesh.sample_points_uniformly(number_of_points=noOfSamplingPoints)
    if(args.use_normals):
        pcd.estimate_normals()
    points = torch.tensor(pcd.points).float()
    points = preprocess_points(points)
    print(f'Points shape: {points.shape}')
    return points


def generateEncodingAndSave(classifier, points, meshPath):
    with torch.no_grad():
        _,_, encoding = classifier(points)
        print(f'Encoding: {encoding}')
        print(f'Encoding shape: {encoding.shape}')

        fileNameToWrite = os.path.splitext(meshPath)[0] + '.npy'
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
    parser.add_argument('--data_path', type=str, required=True, default='data/myshapenet/raw_obj/bed/', help='give data path')
    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args = parse_args()
    checkpointPath = 'log/classification/pointnet2_ssg_wo_normals/checkpoints/best_model.pth'
    noOfSamplingPoints = 1024
    modelName = 'pointnet2_cls_ssg'
    
    ##UMUR##
    #parse data path argument
    dataPath = args.data_path
    meshPaths = readData(dataPath)
    
    '''MODEL LOADING'''
    classifier = loadModel(args, modelName, checkpointPath)
    
    for meshPath in meshPaths:
        '''INPUT'''
        points = getPointCloudFromMeshPath(args, meshPath, noOfSamplingPoints)
        if not args.use_cpu:
            points = points.cuda()
        '''GENERATE ENCODINGS'''
        generateEncodingAndSave(classifier, points, meshPath)
    
if __name__ == '__main__':
    args = parse_args()
    main(args)


