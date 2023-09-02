scriptDir=$(cd $(dirname $0); pwd)

dir=${scriptDir}/logs
if [ ! -d "$dir" ];then
mkdir $dir 
touch $dir/log.txt
fi

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 \
tools/train_net.py --config=${scriptDir}/config.yaml --multi_gpu 

