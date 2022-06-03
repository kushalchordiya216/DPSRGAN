# DPSRGAN: Dilation Patch Super-Resolution Generative Adversarial Networks

A PyTorch implementation of [DPSRGAN: Dilation Patch Super-Resolution Generative Adversarial Networks](https://ieeexplore.ieee.org/document/9417903)

## Usage
### Installation
```bash
git clone https://github.com/kushalchordiya216/Super-Resolution.git
cd Super-Resolution
pip3 install -r requirements.txt
```

### Training
To pretrain the generator before GAN training:
```bash
python3 train.py --data_dir <path to HR images> --network SRResNet
```

For GAN training:
```bash
python3 train.py --data_dir <path to HR images> --network SRGAN --pretrain_gen <path to pretrained generator model file>
```
To view more argument descriptions:
```bash
python3 train.py --help
```

### Testing
This will save the predicted images in the directory `./preds/`
```bash
python3 --model_path <path to pretrained generator> --data_dir <path to directory containing LR images>
```
