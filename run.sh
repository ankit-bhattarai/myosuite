# python ArmReach_LLC_UitB.py --experiment_id no_vision_no_adaptive_higher_batch_size_2 --project_id myosuite_myouser_pointing --batch_size 1024 --num_envs 12288 --num_minibatches 24 --no-vision --adaptive_increase_success_rate 1.1 --adaptive_decrease_success_rate -0.1 --init_target_area_width_scale 1.0 --n_train_steps 40_000_000


# python ArmReach_LLC_UitB.py --experiment_id no_vision_no_adaptive_higher_batch_size_bigger_network --project_id myosuite_myouser_pointing --batch_size 1024 --num_envs 12288 --num_minibatches 24 --no-vision --adaptive_increase_success_rate 1.1 --adaptive_decrease_success_rate -0.1 --init_target_area_width_scale 1.0 --n_train_steps 25_000_000 --policy_hidden_layer_sizes 512 512 512 --value_hidden_layer_sizes 512 512 512


# python ArmReach_LLC_UitB.py --experiment_id no_vision_no_adaptive --project_id myosuite_myouser_pointing --batch_size 256 --num_envs 3072 --num_minibatches 24 --no-vision --adaptive_increase_success_rate 1.1 --adaptive_decrease_success_rate -0.1 --init_target_area_width_scale 1.0 --n_train_steps 25_000_000

python ArmReach_LLC_UitB.py --experiment_id depth_only_vision_bottlenecked_with_autoencoder_and_reconstruction_loss_w_adaptive_better_terms --project_id testing --batch_size 128 --num_envs 1024 --num_minibatches 8 --vision_mode depth --n_train_steps 75_000_000 --proprioception_output_size 48 --vision_output_size 48 --fused_output_size 128 --reconstruction_loss_weight 0.5 --adaptive_increase_success_rate 0.6 --adaptive_decrease_success_rate 0.3
#--reach_metric_coefficient 10.0 --restore_params_path myosuite-mjx-policies/depth_only_vision_bottlenecked_no_adaptive_params


bash subsequent_runs.sh
# options for reach_metric_coefficent: 9, 8, 5
# should try to double num_envs and thus num_minibatches - cant be done at all because of memory constraints
# default weights_reach = 1.0, weights_bonus = 8.0, reach_metric_coefficient = 10.0
# should retry to train the just trained policy with higher weights_reach and weights_bonus
# and should try to see whether the weight coefficients have any significant effect on the performance