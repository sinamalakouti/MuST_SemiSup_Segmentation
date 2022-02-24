
CUDA_VISIBLE_DEVICES=1 python3 Pgs3_WMH.py \
    --output_dir=../miccai2022_test/anthony \
    --config=configs_pgs3/wmh-semi_alternate-5-41-0.5.yaml \
    > ../miccai2022_test/anthony/out.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python3 Pgs3_WMH.py \
    --output_dir=../miccai2022_test/anthony \
    --config=configs_pgs3/wmh-semi_alternate-5-42-0.5.yaml \
    > ../miccai2022_test/anthony/out.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python3 Pgs3_WMH.py \
    --output_dir=../miccai2022_test/anthony \
    --config=configs_pgs3/wmh-semi_alternate-5-40-0.5.yaml \
    > ../miccai2022_test/anthony/out.txt 2>&1



CUDA_VISIBLE_DEVICES=1 python3 Pgs3_WMH.py \
    --output_dir=../miccai2022_test/anthony \
    --config=configs_pgs3/wmh-semi_alternate-5-41-0.8.yaml \
    > ../miccai2022_test/anthony/out.txt 2>&1



CUDA_VISIBLE_DEVICES=1 python3 Pgs3_WMH.py \
    --output_dir=../miccai2022_test/anthony \
    --config=configs_pgs3/wmh-semi_alternate-5-42-0.8.yaml \
    > ../miccai2022_test/anthony/out.txt 2>&1


CUDA_VISIBLE_DEVICES=1 python3 Pgs3_WMH.py \
    --output_dir=../miccai2022_test/anthony \
    --config=configs_pgs3/wmh-semi_alternate-5-43-0.8.yaml \
    > ../miccai2022_test/anthony/out.txt 2>&1
