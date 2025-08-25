#!/bin/bash
# sleep infinity

echo '**********************ENV*************************'
module load Python/3.12.3-GCCcore-13.3.0
module load bzip2
source /home/zxd283/dgzhang_new/zdg_med/bin/activate
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


echo '**********************cuda:0 cifar*************************'
echo '**********************cuda:1 cifar*************************'
echo '**********************cuda:2 imagenette*************************'
echo '**********************cuda:3 imagenette*************************'
