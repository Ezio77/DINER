
# CUDA_VISIBLE_DEVICES=1 python train_different_hash_dimensions.py -c config/different_hash_dimensions.ini

# CUDA_VISIBLE_DEVICES=1 python train_interpolate.py -c config/interpolate.ini

# CUDA_VISIBLE_DEVICES=1 python train_two_dim.py -c config/two_dim.ini

# CUDA_VISIBLE_DEVICES=0 python train_lin.py -c config/lin.ini

CUDA_VISIBLE_DEVICES=2 python train_interp_after_training.py -c config/interpolate.ini