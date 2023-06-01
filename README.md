# Task-Oriented Multi-Objective Optimization

This repository contains the code for the paper: Generating Adversarial Examples with Task Oriented Multi-Objective Optimization, accepted to TMLR 2023.
Link to the paper: https://openreview.net/forum?id=2f81Q622ww


## Getting pre-trained models
To get pretrained models (i.e., with adversarial training), please run the script below: 
```
    bash run_train.sh
```

For manual training, there are some parameters that need to be considered: 
- `arch`: architecture code name which can be {`resnet18`, `vgg16`, `efficientnet`, `googlenet`, `wideresnet`}. Further information can be found in `utils_arch.py`. 
- `num_models`: number of ensemble members, for examples, if `arch==resnet18` and `num_models==3` which refers to an ensemble of three resnet18 models. The default setting is num_models=1 which means that it is a single model. 
- `dataset`: training dataset, either cifar10 or cifar100. 
- `method`: type of training method (i.e., trainer which has been defined in `trainer.py`). For examples, `ADV_Trainer` refers to adversarial training method, which is the default setting. 

## Evaluating attacks 
To get results of three main attack settings (ENS, EoT and UNI), please run the following scripts: 

ENS - Generating Attack for Ensemble of Models. 
```
    bash run_ensemble.sh
```

EoT - Generating Robust Attack against Ensemble of Transformations.  
```
    bash run_eot.sh
```

UNI - Generating Universal perturbations.  
```
    bash run_uniper.sh
```

To get results on evaluating the transferability of adverasrial examples we need to run in two phases. 
The first phase is to generate and save adversarial examples to a checkpoint file using the script below:
```
    bash run_genadv.sh
```

The second phase is to load the adversarial examples to attack other models using the script below: 
```
    bash run_transferability.sh
```


To get results of adversarial training, please run the following script: 
```
    bash run_adv_train.sh
```

To get results on attacking ImageNet dataset with Robustbench pretrained models, please run the following script: 
```
    bash run_ens_imagenet.sh
    bash run_eot_imagenet.sh
```

## Requirements 
- advertorch==0.2.3
- robustbench

## Citation 
If you find this repository useful, please cite our paper: 
```
@article{bui2023generating,
title={Generating Adversarial Examples with Task Oriented Multi-Objective Optimization},
author={Anh Tuan Bui and Trung Le and He Zhao and Quan Hung Tran and Paul Montague and Dinh Phung},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=2f81Q622ww},
note={}
}
```
