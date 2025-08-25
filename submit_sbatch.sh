#!/usr/bin/bash
# ------------------------ Config ------------------------ #
job_name=$1        # 任务名称
account="aiscii"               # 账户名称    aiscii(h200) sxb1592_aisc(a100 84g)
partition="aiscii" # 分区名称   aiscii aisc
nnodes=1                      # 所需节点数
gpus_per_node=1               # 每个节点所需的 GPU 数量
cpus_per_task=32               # 每个 GPU 所需的 CPU 数量
output_dir="slurm-outputs"    # 输出目录
mem="512gb"                # 内存大小
time="32:00:00"             # 任务运行时间
# ------------------------ Setup ------------------------ #
timestamp=$(date +%Y%m%d_%H%M%S)
output_dir=${output_dir}/${timestamp}
export TIMESTAMP=${timestamp}
export OUTPUT_DIR=${output_dir}
# mkdir -p ${output_dir}
# ------------------------ Submit ------------------------ #
sbatch \
    --job-name=${job_name} \
    --account=${account} \
    --partition=${partition} \
    --nodes=${nnodes} \
    --gres=gpu:${gpus_per_node} \
    --cpus-per-task=${cpus_per_task} \
    --output=./${timestamp}_slurm-%j.log \
    --mem=${mem} \
    --time=${time} \
    $1