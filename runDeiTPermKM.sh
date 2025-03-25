torchrun --standalone --nproc_per_node=4 trainPerm.py \
    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
    --sparsityType permkm \
    --nm_n 1 \
    --nm_m 20 \
    --mlp_layer MaskedMLP \
    --sparsity 0.95 \
    --experiment deit-tiny-permnm_120_I100_patchLinear_permStruc_0.95 \
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

torchrun --standalone --nproc_per_node=4 trainPerm.py \
    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
    --sparsityType permkm \
    --nm_n 2 \
    --nm_m 20 \
    --mlp_layer MaskedMLP \
    --sparsity 0.90 \
    --experiment deit-tiny-permnm_220_I100_patchLinear_permStruc_0.90 \
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

torchrun --standalone --nproc_per_node=4 trainPerm.py \
    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
    --sparsityType permkm \
    --nm_n 4 \
    --nm_m 20 \
    --mlp_layer MaskedMLP \
    --sparsity 0.80 \
    --experiment deit-tiny-permnm_420_I100_patchLinear_permStruc_0.80 \
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

torchrun --standalone --nproc_per_node=4 trainPerm.py \
    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
    --sparsityType permkm \
    --nm_n 2 \
    --nm_m 40 \
    --mlp_layer MaskedMLP \
    --sparsity 0.95 \
    --experiment deit-tiny-permnm_240_I100_patchLinear_permStruc_0.95 \
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

torchrun --standalone --nproc_per_node=4 trainPerm.py \
    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
    --sparsityType permkm \
    --nm_n 4 \
    --nm_m 40 \
    --mlp_layer MaskedMLP \
    --sparsity 0.90 \
    --experiment deit-tiny-permnm_440_I100_patchLinear_permStruc_0.90 \
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

torchrun --standalone --nproc_per_node=4 trainPerm.py \
    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
    --sparsityType permkm \
    --nm_n 8 \
    --nm_m 40 \
    --mlp_layer MaskedMLP \
    --sparsity 0.80 \
    --experiment deit-tiny-permnm_840_I100_patchLinear_permStruc_0.80 \
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
