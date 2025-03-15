torchrun --standalone --nproc_per_node=2 trainPerm.py \
    /p/dataset/ImageNet100/ \
    --sparsityType km \
    --sparsity 0.90 \
    --experiment deit-tiny-kmI100_patchLinear_permStruc_4090_0.90 \
    --model deit_tiny_patch16_224 \
    --batch-size 512 \
    --epochs 300 \
    --opt adamw \
    --lr 1e-3 \
    --weight-decay 0.005 \
    --amp \
    --log-wandb \
    --channels-last \
    --pin-mem \
    --workers 16 \
    --log-interval 100

torchrun --standalone --nproc_per_node=2 trainPerm.py \
    /p/dataset/ImageNet100/ \
    --sparsityType permkm \
    --sparsity 0.90 \
    --experiment deit-tiny-permkmI100_patchLinear_permStruc_4090_0.90 \
    --model deit_tiny_patch16_224 \
    --batch-size 512 \
    --epochs 300 \
    --opt adamw \
    --lr 1e-3 \
    --weight-decay 0.005 \
    --amp \
    --log-wandb \
    --channels-last \
    --pin-mem \
    --workers 16 \
    --log-interval 100