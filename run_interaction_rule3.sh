
# ========== SMD ==========

top_m3
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smd/machine-1-1_v2/top_m3 --name smd_machine-1-1_v2_top_m3

bottom_m3
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smd/machine-1-1_v2/bottom_m3 --name smd_machine-1-1_v2_bottom_m3

top_m5
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smd/machine-1-1_v2/top_m5 --name smd_machine-1-1_v2_top_m5

bottom_m5
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.005 --test_stride 1 --n_block 3 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smd/machine-1-1_v2/bottom_m5 --name smd_machine-1-1_v2_bottom_m5

# ========== MSL ==========

# top_m3
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.002 --test_stride 1 --n_block 5 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/msl/P-15_v2/top_m3 --name msl_P-15_v2_top_m3

bottom_m3
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.002 --test_stride 1 --n_block 5 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/msl/P-15_v2/bottom_m3 --name msl_P-15_v2_bottom_m3

top_m5
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.002 --test_stride 1 --n_block 5 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/msl/P-15_v2/top_m5 --name msl_P-15_v2_top_m5

bottom_m5
python main.py --seq_len 30 --pred_len 1 --pd_beta 0 --sample_p 0.2 --sparse_th 0.002 --test_stride 1 --n_block 5 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/msl/P-15_v2/bottom_m5 --name msl_P-15_v2_bottom_m5

# ========== SMAP ==========

# top_m3
python main.py --seq_len 70 --pred_len 1 --pd_beta 1 --sample_p 0.2 --sparse_th 0.008 --test_stride 1 --n_block 6 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smap/T-3_v2/top_m3 --name smap_T-3_v2_top_m3

# bottom_m3
python main.py --seq_len 70 --pred_len 1 --pd_beta 1 --sample_p 0.2 --sparse_th 0.008 --test_stride 1 --n_block 6 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smap/T-3_v2/bottom_m3 --name smap_T-3_v2_bottom_m3

# top_m5
python main.py --seq_len 70 --pred_len 1 --pd_beta 1 --sample_p 0.2 --sparse_th 0.008 --test_stride 1 --n_block 6 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smap/T-3_v2/top_m5 --name smap_T-3_v2_top_m5

# bottom_m5
python main.py --seq_len 70 --pred_len 1 --pd_beta 1 --sample_p 0.2 --sparse_th 0.008 --test_stride 1 --n_block 6 --ff_dim 1024 --dropout 0 --learning_rate 0.0001 --device cpu --data ./datasets/smap/T-3_v2/bottom_m5 --name smap_T-3_v2_bottom_m5