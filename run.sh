python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3 --depth 4 \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone convnext_base --pretrained --timestamp 'convnext-model-pretrained-unfreezed'

python main.py                       \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 128 \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 9 \
    --dim 384 --patch-size 8         \
    --backbone patch --timestamp 'patch-model'

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone densenet121 --pretrained --timestamp 'densenet-model-pretrained-unfreezed'

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone efficientnet --pretrained --timestamp 'efficientnet-model-pretrained-unfreezed'

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 128  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --freeze-backbone \
    --backbone convnext_base --pretrained --timestamp 'convnext-model-pretrained-freezed'

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 2e-5    \
    --input-size 480 --kernel-size 1 --only-fc \
    --backbone convnext_base --pretrained --timestamp 'convnext-fc-pretrained'
