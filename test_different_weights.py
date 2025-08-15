import numpy as np
import subprocess

values = np.logspace(np.log10(0.1), np.log10(10), num=5)

combos = [(dist, xw) for dist in values for xw in values]
batch_size = 1

def run_batch(commands):
    procs = [subprocess.Popen(cmd) for cmd in commands]
    for p in procs:
        p.wait()

for i in range(0, len(combos), batch_size):
    batch = combos[i:i+batch_size]
    cmds = []
    for dist_coef, x_weight in batch:
        dist_str = f"{dist_coef:.3f}".replace('.', '_')
        xw_str = f"{x_weight:.3f}".replace('.', '_')
        experiment_id = f"steering_w_dist_{dist_str}_xweight_{xw_str}"

        cmd = [
            "python", "ArmReach_steering.py",
            "--experiment_id", experiment_id,
            "--project_id", "steeringv3",
            "--batch_size", "256",
            "--num_envs", "3072",
            "--num_minibatches", "12",
            "--n_train_steps", "0",
            "--no-vision",
            "--episode_length", "400",
            "--distance_reach_metric_coefficient", f"{dist_coef:.3f}",
            "--x_reach_weight", f"{x_weight:.3f}",
            "--success_bonus", "10.0",
            "--phase_0_to_1_transition_bonus", "0.0",
            "--policy_hidden_layer_sizes", "128", "128", "128", "128",
            "--value_hidden_layer_sizes", "256", "256", "256", "256", "256",
            "--n_eval_eps", "10",
            "--screen_friction", "0.05",
            "--restore_params_path", "myosuite-mjx-policies/" + experiment_id + "_params",
            "--no_wandb"
        ]
        cmds.append(cmd)

    print(f"Starte Batch mit {len(cmds)} Jobs...")
    run_batch(cmds)
    print("Batch fertig.")
