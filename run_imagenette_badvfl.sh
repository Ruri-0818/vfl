#!/bin/bash
# sleep infinity

echo '**********************ENV*************************'
module load Python/3.12.3-GCCcore-13.3.0
module load bzip2
source /home/zxd283/zdg_fed/fed/bin/activate
export WANDB_API_KEY=cc7cc69bd9cc3e0c3e1de12afdbb0d911b11ae91     # dgzhang
n_gpus=$(( $(echo "$SLURM_JOB_GPUS" | tr -cd ',' | wc -c) + 1 )) # 根据环境变量获取 GPU 数量
export MASTER_ADDR=$SLURMD_NODENAME          # 获取主节点地址
export MASTER_PORT=$((RANDOM % 101 + 20000)) # 随机生成通信端口，防止端口冲突，这里使用 20000-20100 之间的随机数，可按需修改
export NNODES=$SLURM_JOB_NUM_NODES           # 获取节点数
export NPROC_PER_NODE=${n_gpus}              # 获取每个节点的 GPU 数量
export PYTHONPATH=$(pwd):$PYTHONPATH
export NODE_RANK=${NODE_RANK:-$SLURM_NODEID} # 获取当前节点的编号
export OMP_NUM_THREADS=$NPROC_PER_NODE
# pip install -r /home/zxd283/dgzhang_new/code/requirements.txt


cd /home/zxd283/zdg_fed/fed_code
echo '**********************imagenette badvfl*************************'
# None
nohup python train_cifar_badvfl_with_inference.py --dataset IMAGENETTE --batch-size 32 --epochs 50 --lr 0.0005 --party-num 4 --bkd-adversary 1 --target-class 0 --trigger-size 5 --backdoor-weight 2.0 --has-label-knowledge False --checkpoint-dir "./res/checkpoints_imagenette_badvfl_none" --data-dir "/home/zxd283/zdg_fed/datasets/imagenette2-320" --device cuda:0 --Ebkd 5 --poison-budget 0.7 --trigger-intensity 1.0 --num-classes 10 --position dr --trigger-type pattern --inference-weight 0.1 --defense-type NONE --confidence-threshold 0.3 > ./res/imagenette_badvfl_none.log 2>&1 &
# Ours 
nohup python train_cifar_badvfl_with_inference.py --dataset IMAGENETTE --batch-size 32 --epochs 50 --lr 0.0005 --party-num 4 --bkd-adversary 1 --target-class 0 --trigger-size 5 --backdoor-weight 2.0 --has-label-knowledge False --checkpoint-dir "./res/checkpoints_imagenette_badvfl_ours" --data-dir "/home/zxd283/zdg_fed/datasets/imagenette2-320" --device cuda:0 --Ebkd 5 --poison-budget 0.7 --trigger-intensity 1.0 --num-classes 10 --position dr --trigger-type pattern --inference-weight 0.1 --defense-type MY --confidence-threshold 0.3 --tau 3.6 --k-min 2 > ./res/imagenette_badvfl_ours.log 2>&1 &
# DPSGD
nohup python train_cifar_badvfl_with_inference.py --dataset IMAGENETTE --batch-size 32 --epochs 50 --lr 0.0005 --party-num 4 --bkd-adversary 1 --target-class 0 --trigger-size 5 --backdoor-weight 0.8 --has-label-knowledge False --checkpoint-dir ./res/checkpoints_imagenette_badvfl_DPSGD --data-dir "/home/zxd283/zdg_fed/datasets/imagenette2-320" --device cuda:0 --Ebkd 5 --poison-budget 0.7 --trigger-intensity 1.0 --num-classes 10 --position dr --trigger-type pattern --inference-weight 0.1 --defense-type DPSGD --confidence-threshold 0.3 --dpsgd-noise-multiplier 1.0 --dpsgd-max-grad-norm 0.8 --dpsgd-epsilon 3.0 --patience 10 --min-epochs 20 --momentum 0.9 --weight-decay 5e-4 > ./res/imagenette_badvfl_dpsgd.log 2>&1 &
# MP
nohup python train_cifar_badvfl_with_inference.py --dataset IMAGENETTE --batch-size 32 --epochs 50 --lr 0.0005 --party-num 4 --bkd-adversary 1 --target-class 0 --trigger-size 5 --backdoor-weight 0.8 --has-label-knowledge False --checkpoint-dir ./res/checkpoints_imagenette_badvfl_MP --data-dir "/home/zxd283/zdg_fed/datasets/imagenette2-320" --device cuda:0 --Ebkd 5 --poison-budget 0.7 --trigger-intensity 1.0 --num-classes 10 --position dr --trigger-type pattern --inference-weight 0.1 --defense-type MP --confidence-threshold 0.3 --mp-pruning-amount 0.3 --patience 10 --min-epochs 20 --momentum 0.9 --weight-decay 5e-4 > ./res/imagenette_badvfl_mp.log 2>&1 &
# ANP
nohup python train_cifar_badvfl_with_inference.py --dataset IMAGENETTE --batch-size 32 --epochs 50 --lr 0.0005 --party-num 4 --bkd-adversary 1 --target-class 0 --trigger-size 5 --backdoor-weight 1.2 --has-label-knowledge False --checkpoint-dir ./res/checkpoints_imagenette_badvfl_ANP --data-dir "/home/zxd283/zdg_fed/datasets/imagenette2-320" --device cuda:0 --Ebkd 5 --poison-budget 0.7 --trigger-intensity 1.0 --num-classes 10 --position dr --trigger-type pattern --inference-weight 0.1 --defense-type ANP --anp-sigma 0.2 --confidence-threshold 0.3 > ./res/imagenette_badvfl_anp.log 2>&1 &
# BDT
nohup python train_cifar_badvfl_with_inference.py --dataset IMAGENETTE --batch-size 32 --epochs 50 --lr 0.0005 --party-num 4 --bkd-adversary 1 --target-class 0 --trigger-size 5 --backdoor-weight 0.8 --has-label-knowledge False --checkpoint-dir ./res/checkpoints_imagenette_badvfl_BDT --data-dir "/home/zxd283/zdg_fed/datasets/imagenette2-320" --device cuda:0 --Ebkd 5 --poison-budget 0.7 --trigger-intensity 1.0 --num-classes 10 --position dr --trigger-type pattern --inference-weight 0.1 --defense-type BDT --confidence-threshold 0.3 --bdt-prune-ratio 0.3 --patience 10 --min-epochs 20 --momentum 0.9 --weight-decay 5e-4 > ./res/imagenette_badvfl_bdt.log 2>&1 &
# VFLIP
nohup python train_cifar_badvfl_with_inference.py --dataset IMAGENETTE --batch-size 32 --epochs 50 --lr 0.0005 --party-num 4 --bkd-adversary 1 --target-class 0 --trigger-size 5 --backdoor-weight 0.8 --has-label-knowledge False --checkpoint-dir ./res/checkpoints_imagenette_badvfl_VFLIP --data-dir "/home/zxd283/zdg_fed/datasets/imagenette2-320" --device cuda:0 --Ebkd 5 --poison-budget 0.7 --trigger-intensity 1.0 --num-classes 10 --position dr --trigger-type pattern --inference-weight 0.1 --defense-type VFLIP --confidence-threshold 0.3 --vflip-threshold 2.0 --patience 10 --min-epochs 20 --momentum 0.9 --weight-decay 5e-4 --input-dim 1024 > ./res/imagenette_badvfl_vflip.log 2>&1 &
# ISO
nohup python train_cifar_badvfl_with_inference.py --dataset IMAGENETTE --batch-size 32 --epochs 50 --lr 0.0005 --party-num 4 --bkd-adversary 1 --target-class 0 --trigger-size 5 --backdoor-weight 2.0 --has-label-knowledge False --checkpoint-dir ./res/checkpoints_imagenette_badvfl_ISO --data-dir "/home/zxd283/zdg_fed/datasets/imagenette2-320" --device cuda:0 --Ebkd 5 --poison-budget 0.7 --trigger-intensity 1.0 --num-classes 10 --position dr --trigger-type pattern --inference-weight 0.1 --defense-type ISO --iso-lr 0.001 --confidence-threshold 0.3 --input-dim 1024 > ./res/imagenette_badvfl_iso.log 2>&1 &
# 等待后台结束
wait