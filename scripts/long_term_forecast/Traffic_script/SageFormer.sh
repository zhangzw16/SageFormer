# export CUDA_VISIBLE_DEVICES=0

model_name=SageFormer
seq_len=96
cls_len=1
graph_depth=3
knn=16
e_layers=3

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len"_96_el"$e_layers"_cls"$cls_len"_gdep"$graph_depth"_k"$knn \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --lradj 'type3' \
  --train_epochs 20 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id "traffic_"$seq_len"_192_el"$e_layers"_cls"$cls_len"_gdep"$graph_depth"_k"$knn \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --lradj 'type3' \
  --train_epochs 20 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id "traffic_"$seq_len"_336_el"$e_layers"_cls"$cls_len"_gdep"$graph_depth"_k"$knn \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --lradj 'type3' \
  --train_epochs 20 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id "traffic_"$seq_len"_720_el"$e_layers"_cls"$cls_len"_gdep"$graph_depth"_k"$knn \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --lradj 'type3' \
  --train_epochs 20 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --itr 1
