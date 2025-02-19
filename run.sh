torchrun --standalone --nproc_per_node=4 trainMask.py \
    /scratch/atyagi2/imageNet \
    --model mixer_square_s16_224 \
    --batch-size 512 \
    --epochs 200 \
    --opt adamw \
    --lr 1e-3 \
    --weight-decay 0.005 \
    --amp
