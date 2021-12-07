#!/bin/sh
for bsz in 96 32 64;
do
	for alp in 0.999 0.99 0.9;
	do
		for lnum in 4 5 6;
		do
		# python train.py --dataset 'mydataset' --exp Pretrain --gpu 0 --moco --nce-k 16384 --batch-size $bsz --alpha $alp --num-layer $lnum;
		echo '1';
		# python generate.py --gpu 0 --dataset 'mydataset' --load-path "./Pretrain_moco_True_mydataset_gin_layer_${lnum}_lr_0.005_decay_1e-05_bsz_${bsz}_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_${alp}_r0.3/current.pth";
		echo '2';
		python gcc/tasks/graph_classification.py --dataset "mydataset" --hidden-size 2 --model from_numpy_graph --emb-path "Pretrain_moco_True_mydataset_gin_layer_${lnum}_lr_0.005_decay_1e-05_bsz_${bsz}_hid_64_samples_2000_nce_t_0.07_nce_k_16384_rw_hops_256_restart_prob_0.8_aug_1st_ft_False_deg_16_pos_32_momentum_${alp}_r0.3/mydataset.npy";
		echo '3';
		done	
	done
done

