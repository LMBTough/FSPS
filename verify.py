"""Implementation of evaluate attack result."""
import os
import csv
import random
import numpy as np
import torch
import argparse
from torch.autograd import Variable as V
from torch import nn
# from torch.autograd.gradcheck import zero_gradients
from torchvision import transforms as T
from Normalize import Normalize, TfNormalize
from loader import ImageNet
from torch.utils.data import DataLoader
from torch_nets import (
    tf_inception_v3,
    tf_inception_v4,
    tf_resnet_v2_50,
    tf_resnet_v2_101,
    tf_resnet_v2_152,
    tf_inc_res_v2,
    tf_adv_inception_v3,
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='', help='FGSM methods: DI,TI,MI')
parser.add_argument('--input_csv', type=str, default='./dataset/images.csv', help='Input directory with images.')
parser.add_argument('--output_dir', type=str, default='./outputs/', help='Output directory with adversarial images.')
parser.add_argument('--output_csv', type=str, default='result.csv', help='Output csv.')
parser.add_argument("--num_images", type=int, default=1000, help="How many images load at one time.")
opt = parser.parse_args()
batch_size = 10

input_csv = opt.input_csv
input_dir = './dataset/images/images'
adv_dir = opt.output_dir


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# Define a function to set the seed for reproducibility
def set_seed(seed):
    # Set random seed for torch CPU
    torch.manual_seed(seed)
    # Set random seed for all GPUs
    torch.cuda.manual_seed_all(seed)
    # Set random seed for numpy
    np.random.seed(seed)
    # Set deterministic mode for CuDNN to True
    torch.backends.cudnn.deterministic = True
    # Disable CuDNN benchmarking to ensure deterministic results
    torch.backends.cudnn.benchmark = False
    # Set random seed for Python hash
    random.seed(seed)
    # Set the PYTHONHASHSEED environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)
set_seed(2023)

def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        TfNormalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),)
    return model

def verify(model_name, path):

    model = get_model(model_name, path)

    X = ImageNet(adv_dir, input_csv, T.Compose([T.ToTensor()]), num_images=opt.num_images)
    data_loader = DataLoader(X, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    sum = 0
    for images, _, gt_cpu in data_loader:
        gt = gt_cpu.cuda()
        images = images.cuda()
        with torch.no_grad():
            sum += (model(images)[0].argmax(1) != (gt+1)).detach().sum().cpu()

    print(model_name + '  acu = {:.2%}'.format(sum / opt.num_images))
    print("===================================================")
    result = sum.item() / opt.num_images
    return result

def main():

    model_names = ['tf_inception_v3','tf_inception_v4','tf_inc_res_v2','tf_resnet_v2_50','tf_resnet_v2_101','tf_resnet_v2_152','tf_ens3_adv_inc_v3','tf_ens4_adv_inc_v3','tf_ens_adv_inc_res_v2']

    models_path = './models/'
    # for model_name in model_names:
    #     result = verify(model_name, models_path)
        


    # 将数据存储到二维列表中
    data = [[model_name, verify(model_name, models_path), opt.method] for model_name in model_names]

    # 将数据写入到 CSV 文件中
    with open(opt.output_csv, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model Name', 'Success Rate','Method'])
        writer.writerows(data)

    print("The results have been successfully written to the "+ opt.output_csv)


if __name__ == '__main__':
    main()