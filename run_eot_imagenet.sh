# TARGETED ATTACK 
# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --loss_type=cw --batch=50 --Trf=TrfE --method=PGD_Linf_EoT --num_btest=100 --eotsto=False  --targeted=True --inf=cw_det_targeted=True

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --loss_type=cw --batch=50 --Trf=TrfE --method=MinMaxEoT --num_btest=100 --eotsto=False  --targeted=True --inf=cw_det_targeted=True

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --moo_lr=0.005 --loss_type=cw --batch=50 --Trf=TrfE --method=MOOEoTv42 --num_btest=100 --eotsto=False  --targeted=True --inf=cw_det_targeted=True

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20 --loss_type=cw --batch=50 --Trf=TrfE --method=MOOTOEoTv42 --num_btest=100  --m1=50.0 --moo_lr=0.005 --eotsto=False --targeted=True --inf=cw_det_targeted=True


# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --loss_type=cw --batch=50 --Trf=TrfE --method=PGD_Linf_EoT --num_btest=100 --eotsto=True  --targeted=True --inf=cw_sto_targeted=True

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --loss_type=cw --batch=50 --Trf=TrfE --method=MinMaxEoT --num_btest=100 --eotsto=True  --targeted=True --inf=cw_sto_targeted=True

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --moo_lr=0.005 --loss_type=cw --batch=50 --Trf=TrfE --method=MOOEoTv42 --num_btest=100 --eotsto=True --targeted=True --inf=cw_sto_targeted=True

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20 --loss_type=cw --batch=50 --Trf=TrfE --method=MOOTOEoTv42 --num_btest=100  --m1=50.0 --moo_lr=0.005 --eotsto=True --targeted=True --inf=cw_sto_targeted=True

## UNTAR ATTACK 
# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --loss_type=cw --batch=50 --Trf=TrfE --method=PGD_Linf_EoT --num_btest=100 --eotsto=False  --targeted=False --inf=cw_det_targeted=False

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --loss_type=cw --batch=50 --Trf=TrfE --method=MinMaxEoT --num_btest=100 --eotsto=False  --targeted=False --inf=cw_det_targeted=False

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --moo_lr=0.005 --loss_type=cw --batch=50 --Trf=TrfE --method=MOOEoTv42 --num_btest=100 --eotsto=False  --targeted=False --inf=cw_det_targeted=False

# python eval_robustbench_eot.py --arch=Salman2020Do_R18 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20 --loss_type=cw --batch=50 --Trf=TrfE --method=MOOTOEoTv42 --num_btest=100  --m1=50.0 --moo_lr=0.005 --eotsto=False --targeted=False --inf=cw_det_targeted=False
