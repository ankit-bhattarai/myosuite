# Steering Task

## Env Setup
To create an environment for steering, do the following

```shell
mamba create -n steering python=3.11 -y
mamba activate steering

mamba install ipykernel ffmpeg -y
pip install "jax[cuda12_local]<0.6.0" mujoco mujoco-mjx brax mediapy hydra-core wandb playground

pip install -e .

```

## Training

To train an agent on the steering task, use the following command

```shell
python ArmReach_steering.py --experiment_id steeringv0 --project_id steering --batch_size 128 --num_envs 1024 --num_minibatches 8 --init_target_area_width_scale 1.0 --vision_mode invalid --adaptive_increase_success_rate 1.1 --adaptive_decrease_success_rate -0.1 --n_train_steps 3_000_000 --no-vision --policy_hidden_layer_sizes 256 256 256 256 256 --value_hidden_layer_sizes 256 256 256 256 256

```


## Evaluation only

To evaluate a trained policy and generate a video use the same script and commands as training but:
- Set the number of train steps `N` in `--n_train_steps N` to `0`.
- Add the flag `--restore_params_path` and set the value to the path of the trained params.
