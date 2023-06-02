# FSPS

This Github repository is the FSPS implementation code

# Requirements
- python 3.8.10
- pytorch 1.13.1
- torchvision 0.14.1
- pretrainedmodels 0.7.0
- numpy 1.21.3
- tqdm 4.63.1


# Setup
Model  | Download Link
------------- | -------------
Inception V3  | [tf2torch_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_inception_v3.npy)
Inception V4| [tf2torch_inception_v4](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_inception_v4.npy)
Inception-ResNet-v2  | [tf2torch_resnet_v2_152](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_inc_res_v2.npy)
ResNet V2 152  | [tf2torch_resnet_v2_152](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_resnet_v2_152.npy)
Inception v3 adv | [tf2torch_adv_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_adv_inception_v3.npy)
Inception ResNet v2 adv  | [adv_inception_resnet_v2_2017_12_18.tar.gz](http://download.tensorflow.org/models/adv_inception_resnet_v2_2017_12_18.tar.gz)
Inception v3 adv ens3  | [tf2torch_ens3_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_ens3_adv_inc_v3.npy)
Inception v3 adv ens4  | [tf2torch_ens4_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_ens4_adv_inc_v3.npy)
Inception ResNet v2 adv ens3  | [tf2torch_ens_adv_inc_res_v2](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf2torch_ens_adv_inc_res_v2.npy)


The models in the table above are from [here](https://github.com/ylhz/tf_to_pytorch_model). These models need to be downloaded and placed under the `models` dir.

# Run
- FSPS

`CUDA_VISIBLE_DEVICES=0 python attack-FSPS.py --output_dir outputs_temp --method TI --num_images 1000 --model inceptionv3`

`CUDA_VISIBLE_DEVICES=0 python attack-FSPS.py --output_dir outputs_temp --method TI --num_images 1000 --model inceptionv4`

`CUDA_VISIBLE_DEVICES=0 python attack-FSPS.py --output_dir outputs_temp --method TI --num_images 1000 --model inceptionresnetv2`

`CUDA_VISIBLE_DEVICES=0 python attack-FSPS.py --output_dir outputs_temp --method TI --num_images 1000 --model resnet152`

- SSA

`CUDA_VISIBLE_DEVICES=0 python attack-SSA.py --output_dir outputs_temp --method DITIMI --num_images 1000 --model inceptionv3`

- Baseline

`CUDA_VISIBLE_DEVICES=0 python attack-baseline.py --output_dir outputs_temp --method DI --num_images 1000 --model inceptionv3`


- verify

`CUDA_VISIBLE_DEVICES=0 python verify.py --method baseline_result_DI-v3 --output_dir outputs_temp/ --num_images 1000 --output_csv result.csv`

# Reference
Code refer to: [SSA](https://github.com/yuyang-long/SSA)