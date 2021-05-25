#!/bin/bash
work_path=$(dirname $0)
python test.py \
    --config $work_path/config.yaml \
    --load-pretrain pretrains/partialconv_input_ch4.pth \
    --load_iter 1000
