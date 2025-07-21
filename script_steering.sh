python ArmReach_steering.py --experiment_id basic_steering --project_id steering --batch_size 128 --num_envs 1024 --num_minibatches 8 --init_target_area_width_scale 1.0 --vision_mode invalid --adaptive_increase_success_rate 1.1 --adaptive_decrease_success_rate -0.1 --n_train_steps 3_000_000 --no-vision --policy_hidden_layer_sizes 256 256 256 256 256 --value_hidden_layer_sizes 256 256 256 256 256

