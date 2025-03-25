#torchrun --standalone --nproc_per_node=4 trainPerm.py \
#    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
#    --sparsityType permBlock \
#    --block_size 16 \
#    --mlp_layer MaskedMLP \
#    --sparsity 0.95 \
#    --experiment deit-tiny-permBlock_16X16_I100_patchLinear_permStruc_0.95 \
#    --model deit_tiny_patch16_224 \
#    --batch-size 512 \
#    --epochs 300 \
#    --opt adamw \
#    --lr 1e-3 \
#    --weight-decay 0.005 \
#    --amp \
#    --log-wandb \
#    --channels-last \
#    --pin-mem \
#    --workers 16 \
#    --log-interval 100
#
#torchrun --standalone --nproc_per_node=4 trainPerm.py \
#    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
#    --sparsityType permBlock \
#    --block_size 16 \
#    --mlp_layer MaskedMLP \
#    --sparsity 0.90 \
#    --experiment deit-tiny-permBlock_16X16_I100_patchLinear_permStruc_0.90 \
#    --model deit_tiny_patch16_224 \
#    --batch-size 512 \
#    --epochs 300 \
#    --opt adamw \
#    --lr 1e-3 \
#    --weight-decay 0.005 \
#    --amp \
#    --log-wandb \
#    --channels-last \
#    --pin-mem \
#    --workers 16 \
#    --log-interval 100
#
#torchrun --standalone --nproc_per_node=4 trainPerm.py \
#    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
#    --sparsityType permBlock \
#    --block_size 16 \
#    --mlp_layer MaskedMLP \
#    --sparsity 0.80 \
#    --experiment deit-tiny-permBlock_16X16_I100_patchLinear_permStruc_0.80 \
#    --model deit_tiny_patch16_224 \
#    --batch-size 512 \
#    --epochs 300 \
#    --opt adamw \
#    --lr 1e-3 \
#    --weight-decay 0.005 \
#    --amp \
#    --log-wandb \
#    --channels-last \
#    --pin-mem \
#    --workers 16 \
#    --log-interval 100

torchrun --standalone --nproc_per_node=4 trainPerm.py \
    /scratch/atyagi2/pytorch-image-models_perm/ImageNet100 \
    --sparsityType permBlock \
    --block_size 32 \
    --mlp_layer MaskedMLP \
    --sparsity 0.95 \
    --experiment deit-tiny-permBlock_32X32_I100_patchLinear_permStruc_0.95 \
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
    --sparsityType permBlock \
    --block_size 32 \
    --mlp_layer MaskedMLP \
    --sparsity 0.90 \
    --experiment deit-tiny-permBlock_32X32_I100_patchLinear_permStruc_0.90 \
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
    --sparsityType permBlock \
    --block_size 32 \
    --mlp_layer MaskedMLP \
    --sparsity 0.80 \
    --experiment deit-tiny-permBlock_32X32_I100_patchLinear_permStruc_0.80 \
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
