#!/usr/bin/env bash

python random_search.py --options="--save_csv,--save_weights_prefix" --label_skip 1 python pyclient_duel.py $* \
  --exploration_rate_end 0.01,0.1 \
  --skip 0,5 \
  --learning_rate 0.01,0.001,0.0001 \
  --discount_rate 0.9,0.99,0.995 \
  --target_rate 1,0.1,0.01,0.001 \
  --hidden_nodes 50,100,200 \
  --hidden_layers 0,1,2 \
  --activation tanh,relu \
  --batch_norm 0,1
