# # Baseline --method=PGD_Linf_ENS
# python eval_robustbench_ens.py --ens_set=robr18r50r50 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20 --loss_type=ce --batch=50 --method=PGD_Linf_ENS --num_btest=100 --targeted=True --inf=ce_targeted

# # Baseline --method=MinMaxEns
# python eval_robustbench_ens.py --ens_set=robr18r50r50 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20 --loss_type=ce --batch=50 --method=MinMaxEns --num_btest=100 --targeted=True --inf=ce_targeted

# # --method=MOOEn 
# python eval_robustbench_ens.py --ens_set=robr18r50r50 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20 --loss_type=ce --batch=50 --method=MOOEns --num_btest=100  --moo_lr=0.005 --targeted=True --inf=ce_targeted

# # --method=MOOTOEns
# python eval_robustbench_ens.py --ens_set=robr18r50r50 --dataset=imagenet --mode=robench --eval_epsilon=0.0156862745 --eval_step_size=0.0039 --eval_num_steps=20  --loss_type=ce --batch=50 --method=MOOTOEns --num_btest=100 --m1=100.0 --moo_lr=0.005 --targeted=True --inf=ce_targeted

