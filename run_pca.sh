
# top_m5
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smd/machine-1-1_v2_pca/top_m5 --name smd_machine-1-1_v2_pca_top_m5

# bottom_m5
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smd/machine-1-1_v2_pca/bottom_m5 --name smd_machine-1-1_v2_pca_bottom_m5

# top_m5
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smd/machine-1-1_yk_v1 --name smd_machine-1-1_yk_v1

# bottom_m5
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smd/machine-1-1_yk_v4 --name smd_machine-1-1_yk_v4
