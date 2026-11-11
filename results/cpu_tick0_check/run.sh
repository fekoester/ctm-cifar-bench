set -e
cd /home/claude/ctm-cifar-bench
COMMON="--device cpu --dataset cifar10 --data_dir /tmp/cifar/data --num_workers 2 --epochs 6 --steps_per_epoch 150 --batch_size 128 --lr 1e-3 --warmup_steps 100 --D 128 --M 8 --H 32 --T_train 5 --Dout_pairs 128 --Dact_pairs 128 --syn_depth 2 --attn_heads 4 --out_root /tmp/cifar/runs --track_every 1000 --log_every 50"
python3 scripts/train_ctm_cifar.py $COMMON --sync_after_update 0 --run_name legacy
python3 scripts/train_ctm_cifar.py $COMMON --sync_after_update 1 --run_name fixed
python3 scripts/train_ctm_cifar.py $COMMON --model_type cnn --eval_rule final --run_name cnn
echo DONE
