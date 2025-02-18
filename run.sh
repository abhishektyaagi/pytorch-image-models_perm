python trainMask.py /p/dataset/imagenet-raw-data \
    --model mixer_square_s16_224 \
    --batch-size 128 \
    --epochs 5 \
    --opt adamw \
    --lr 1e-3 \
    --weight-decay 0.005 \
    --amp
