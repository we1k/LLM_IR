#!/bin/bash

# 设定温度、top_p和种子的值的数组
TEMPERATURES=(0.1 0.3 0.5 0.7 0.9)

# 循环遍历所有值组合
for TEMPERATURE in "${TEMPERATURES[@]}"
do
  echo "Running with Temperature: $TEMPERATURE, Top_p: $TOP_P, Seed: $SEED"
  python3 query_qw_flash_attn.py --temperature $TEMPERATURE --local_run --use-14B
done