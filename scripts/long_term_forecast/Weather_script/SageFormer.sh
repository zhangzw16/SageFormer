# export CUDA_VISIBLE_DEVICES=0

model_name=SageFormer
seq_len=96
cls_len=1
graph_depth=3
knn=16
e_layers=2


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 96 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --n_heads 4 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --lradj 'type3' \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 192 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --n_heads 4 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --lradj 'type3' \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 336 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --n_heads 4 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --lradj 'type3' \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len 720 \
  --e_layers $e_layers \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --n_heads 4 \
  --cls_len $cls_len \
  --graph_depth $graph_depth \
  --knn $knn \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --lradj 'type3' \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 10