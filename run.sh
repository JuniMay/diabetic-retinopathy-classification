python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3 --depth 4 \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone convnext_base --pretrained --timestamp 'convnext-model-pretrained-unfreezed'


python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3 --depth 4 \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone convnext_base --pretrained --timestamp 'convnext-model-pretrained-unfreezed' --valid

python main.py                       \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 128 \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 9 \
    --dim 384 --patch-size 8         \
    --backbone patch --timestamp 'patch-model'


python main.py                       \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 128 \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 9 \
    --dim 384 --patch-size 8         \
    --backbone patch --timestamp 'patch-model' --valid

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone densenet121 --pretrained --timestamp 'densenet-model-pretrained-unfreezed'


python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone densenet121 --pretrained --timestamp 'densenet-model-pretrained-unfreezed' --valid

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone efficientnet --pretrained --timestamp 'efficientnet-model-pretrained-unfreezed'


python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --backbone-lr 2e-5 \
    --backbone efficientnet --pretrained --timestamp 'efficientnet-model-pretrained-unfreezed' --valid

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 128  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --freeze-backbone \
    --backbone convnext_base --pretrained --timestamp 'convnext-model-pretrained-freezed'

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 128  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3    \
    --input-size 480 --kernel-size 1 --freeze-backbone \
    --backbone convnext_base --pretrained --timestamp 'convnext-model-pretrained-freezed' --valid

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 2e-5    \
    --input-size 480 --kernel-size 1 --only-fc \
    --backbone convnext_base --pretrained --timestamp 'convnext-fc-pretrained'

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 2e-5    \
    --input-size 480 --kernel-size 1 --only-fc \
    --backbone convnext_base --pretrained --timestamp 'convnext-fc-pretrained' --valid

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 128  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3 --cabnet --freeze-backbone \
    --input-size 480 \
    --backbone convnext_base --pretrained --timestamp 'cabnet-convnext'
    

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 128  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3 --cabnet --freeze-backbone \
    --input-size 480 \
    --backbone convnext_base --pretrained --timestamp 'cabnet-convnext' --valid

    
python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3 --cabnet --backbone-lr 2e-5 \
    --input-size 480 \
    --backbone convnext_base --pretrained --timestamp 'cabnet-convnext-unfreezed'
    

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3 --cabnet --backbone-lr 2e-5 \
    --input-size 480 \
    --backbone convnext_base --pretrained --timestamp 'cabnet-convnext-unfreezed' --valid

    
    
python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3 --cabnet --cabnet-k 6 --backbone-lr 2e-5 \
    --input-size 480 \
    --backbone convnext_base --pretrained --timestamp 'cabnet-k6-convnext-unfreezed'
    

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3 --cabnet --cabnet-k 6 --backbone-lr 2e-5 \
    --input-size 480 \
    --backbone convnext_base --pretrained --timestamp 'cabnet-k6-convnext-unfreezed' --valid

    
python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/valid' \
    --log-dir 'tf-logs' --lr 1e-3 --cabnet --cabnet-k 4 --backbone-lr 2e-5 \
    --input-size 480 \
    --backbone convnext_base --pretrained --timestamp 'cabnet-k4-convnext-unfreezed'
    

python main.py \
    --num-classes 5 --num-workers 12 \
    --num-epochs 80 --batch-size 32  \
    --valid-data-dir 'data/DDR/test' \
    --log-dir 'tf-logs' --lr 1e-3 --cabnet --cabnet-k 4 --backbone-lr 2e-5 \
    --input-size 480 \
    --backbone convnext_base --pretrained --timestamp 'cabnet-k4-convnext-unfreezed' --valid