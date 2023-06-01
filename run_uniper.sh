
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=4 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=8 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=12 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=16 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=20 --method=PGD_Linf_Uni --inf=ce

# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=4 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=8 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=12 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=16 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=20 --method=MinMaxUni --inf=ce

# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=4 --method=MOOUni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=8 --method=MOOUni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=12 --method=MOOUni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=16 --method=MOOUni --inf=ce
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=20 --method=MOOUni --inf=ce

# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=4 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=8 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=12 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=16 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=resnet18 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=20 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005


## CHANGE ARCHITECTURE 

# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=4 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=8 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=12 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=16 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=20 --method=PGD_Linf_Uni --inf=ce


# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=4 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=8 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=12 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=16 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=20 --method=MinMaxUni --inf=ce

# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=4 --method=MOOUni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=8 --method=MOOUni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=12 --method=MOOUni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=16 --method=MOOUni --inf=ce
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=20 --method=MOOUni --inf=ce

# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=4 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=8 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=12 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=16 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=googlenet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=20 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005

# CHANGE ARCHITECTURE 


# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=4 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=8 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=12 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=16 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=20 --method=PGD_Linf_Uni --inf=ce


# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=4 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=8 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=12 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=16 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=20 --method=MinMaxUni --inf=ce


# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=4 --method=MOOUni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=8 --method=MOOUni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=12 --method=MOOUni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=16 --method=MOOUni --inf=ce
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=20 --method=MOOUni --inf=ce


# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=4 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=8 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=12 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=16 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=efficientnet --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=20 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005

## CHANGE ARCHITECTURE 

# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=4 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=8 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=12 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=16 --method=PGD_Linf_Uni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50  --loss_type=ce --batch=50 --num_K=20 --method=PGD_Linf_Uni --inf=ce


# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=4 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=8 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=12 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=16 --method=MinMaxUni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=50 --loss_type=ce --batch=50 --num_K=20 --method=MinMaxUni --inf=ce


# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=4 --method=MOOUni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=8 --method=MOOUni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=12 --method=MOOUni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=16 --method=MOOUni --inf=ce
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=20 --method=MOOUni --inf=ce


# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=4 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=8 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=12 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=16 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005
# python eval_uni.py --arch=vgg16 --mode=ADV_Trainer --dataset=cifar100 --eval_step_size=0.007 --eval_num_steps=100  --loss_type=ce --batch=50 --num_K=20 --method=MOOTOUni --m1=10.0 --inf=ce_m1=10.0_v4 --num_btest=500 --moo_lr=0.005

