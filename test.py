import os
import cv2

import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np

from pspnet import PSPNet
from imgSegLoader import imgSegDataset

n_classes = 19

models = {
    'squeezenet': lambda: PSPNet(n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    'resnet34': lambda: PSPNet(n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(n_classes=n_classes, sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        print("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


@click.command()
@click.option('--data-path', type=str, help='Path to dataset folder')
@click.option('--output-path', type=str, help='Path to output folder')
@click.option('--backend', type=str, default='resnet34', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--gpu', type=str, default='0', help='List of GPUs for parallel training, e.g. 0,1,2,3')
def test(data_path, output_path, backend, snapshot, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = build_network(snapshot, backend)
    os.makedirs(output_path, exist_ok=True)
    data_path = os.path.abspath(os.path.expanduser(data_path))
    dataset = imgSegDataset()
    dataset.initialize(data_path, 100, False)
    net.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            x, file_name = dataset.test_data(i)
            x = Variable(x).cuda()
            out, out_cls = net(x)
            out_file = os.path.join(output_path, "{}.png".format(file_name))
            print(out_file)
            out_test = out.squeeze().cpu().numpy()
            out = out.max(1)[1].squeeze().cpu().numpy()
            out_img = np.zeros((out.shape[0], out.shape[1], 3))
            body = np.where(out > 0)
            out_img[body[0], body[1], 0] = 255
            out_img[body[0], body[1], 1] = 255
            out_img[body[0], body[1], 2] = 255
            hair = np.where(out == 13)
            out_img[hair[0], hair[1], 0] = 0
            out_img[hair[0], hair[1], 1] = 255
            out_img[hair[0], hair[1], 2] = 0
            
            cv2.imwrite(out_file, out_img)
        
if __name__ == '__main__':
    test()
