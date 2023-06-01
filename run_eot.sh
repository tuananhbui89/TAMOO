## Stochastic CIFAR10
# python eval_eot.py --arch=resnet18 --dataset=cifar10 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=PGD_Linf_EoT --num_btest=500 --eotsto=True  --inf=cw_sto

# python eval_eot.py --arch=resnet18 --dataset=cifar10 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MinMaxEoT --num_btest=500 --eotsto=True  --inf=cw_sto

# python eval_eot.py --arch=resnet18 --dataset=cifar10 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MOOEoTv42 --num_btest=500 --eotsto=True   --moo_lr=0.005 --inf=cw_sto

# python eval_eot.py --arch=resnet18 --dataset=cifar10 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MOOTOEoT --num_btest=500 --m1=100.0 --eotsto=True --moo_lr=0.005 --inf=cw_sto

## Stochastic CIFAR100
# python eval_eot.py --arch=resnet18 --dataset=cifar100 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=PGD_Linf_EoT --num_btest=500 --eotsto=True --inf=cw_sto

# python eval_eot.py --arch=resnet18 --dataset=cifar100 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MinMaxEoT --num_btest=500 --eotsto=True --inf=cw_sto

# python eval_eot.py --arch=resnet18 --dataset=cifar100 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MOOEoTv42 --num_btest=500 --eotsto=True  --moo_lr=0.005 --inf=cw_sto

# python eval_eot.py --arch=resnet18 --dataset=cifar100 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100 --loss_type=cw --batch=100 --Trf=TrfE --method=MOOTOEoT --num_btest=500 --m1=100.0 --moo_lr=0.01 --inf=cw_sto


## Deter - CIFAR10 
# python eval_eot.py --arch=resnet18 --dataset=cifar10 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=PGD_Linf_EoT --num_btest=500 --eotsto=False  --inf=cw_det

# python eval_eot.py --arch=resnet18 --dataset=cifar10 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MinMaxEoT --num_btest=500 --eotsto=False  --inf=cw_det

# python eval_eot.py --arch=resnet18 --dataset=cifar10 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MOOEoTv42 --num_btest=500 --eotsto=False   --moo_lr=0.005 --inf=cw_det

# python eval_eot.py --arch=resnet18 --dataset=cifar10 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100 --loss_type=cw --batch=100 --Trf=TrfE --method=MOOTOEoTv42 --num_btest=500  --m1=100.0 --eotsto=False --moo_lr=0.005 --inf=cw_det


## Deter - CIFAR100
# python eval_eot.py --arch=resnet18 --dataset=cifar100 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=PGD_Linf_EoT --num_btest=500 --eotsto=False  --inf=cw_det

# python eval_eot.py --arch=resnet18 --dataset=cifar100 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MinMaxEoT --num_btest=500 --eotsto=False  --inf=cw_det

# python eval_eot.py --arch=resnet18 --dataset=cifar100 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100  --loss_type=cw --batch=100 --Trf=TrfE --method=MOOEoTv42 --num_btest=500 --eotsto=False   --moo_lr=0.005 --inf=cw_det

# python eval_eot.py --arch=resnet18 --dataset=cifar100 --mode=ADV_Trainer --eval_step_size=0.01 --eval_num_steps=100 --loss_type=cw --batch=100 --Trf=TrfE --method=MOOTOEoTv42 --num_btest=500  --m1=100.0 --eotsto=False --moo_lr=0.005 --inf=cw_det
