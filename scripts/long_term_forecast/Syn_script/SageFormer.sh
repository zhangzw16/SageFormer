export CUDA_VISIBLE_DEVICES=3

model_name=SageFormer
seq_len=96


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/syn_sin \
#   --data_path syn_sin_10.csv \
#   --model_id syn_sin_10_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 10 \
#   --dec_in 10 \
#   --c_out 10 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 256 \
#   --train_epochs 10


# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/syn_sin \
#   --data_path syn_sin_50.csv \
#   --model_id syn_sin_50_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 50 \
#   --dec_in 50 \
#   --c_out 50 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 256 \
#   --train_epochs 10

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/synthetic/syn_sin \
  --data_path syn_sin_100.csv \
  --model_id syn_sin_100_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len 12 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 100 \
  --dec_in 100 \
  --c_out 100 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 128 \
  --train_epochs 10

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/syn_sin \
#   --data_path syn_sin_200.csv \
#   --model_id syn_sin_200_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 200 \
#   --dec_in 200 \
#   --c_out 200 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 64 \
#   --train_epochs 10

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/syn_sin \
#   --data_path syn_sin_500.csv \
#   --model_id syn_sin_500_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 500 \
#   --dec_in 500 \
#   --c_out 500 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 16 \
#   --train_epochs 10

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/syn_sin \
#   --data_path syn_sin_1000.csv \
#   --model_id syn_sin_1000_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 1000 \
#   --dec_in 1000 \
#   --c_out 1000 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 8 \
#   --train_epochs 10

# python -u run.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/synthetic/ \
#   --data_path syn_cycle.csv \
#   --model_id syn_cycle_96_96 \
#   --model $model_name \
#   --data custom \
#   --features M \
#   --seq_len $seq_len \
#   --label_len 18 \
#   --pred_len 12 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --enc_in 10 \
#   --dec_in 10 \
#   --c_out 10 \
#   --des 'Exp' \
#   --itr 1 \
#   --batch_size 32 \
#   --learning_rate 0.0005 \
#   --train_epochs 10
