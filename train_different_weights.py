import numpy as np
import subprocess
import time

values = np.logspace(np.log10(0.1), np.log10(10), num=3)
success_bonus_values = np.linspace(1, 50, num=3)  # z.B. 5 Werte zwischen 1 und 50

# Alle Kombinationen als Liste
combos = [(dist, xw, sb) for dist in values for xw in values for sb in success_bonus_values]

batch_size = 1

def run_batch(commands):
    procs = []
    for cmd in commands:
        p = subprocess.Popen(cmd)
        procs.append(p)
    # Warten, bis alle fertig sind
    for p in procs:
        p.wait()

for i in range(0, len(combos), batch_size):
    batch = combos[i:i+batch_size]

    # cmd1 batch starten und warten
    cmds1 = []
    for dist_coef, x_weight, success_bonus in batch:
        dist_str = f"{dist_coef:.3f}".replace('.', '_')
        xw_str = f"{x_weight:.3f}".replace('.', '_')
        sb_str = f"{success_bonus:.1f}".replace('.', '_')
        experiment_id = f"steering_w_dist_{dist_str}_xweight_{xw_str}_sb_{sb_str}"

        cmd1 = [
            "python", "ArmReach_steering.py",
            "--experiment_id", experiment_id,
            "--project_id", "steeringv3",
            "--batch_size", "256",
            "--num_envs", "3072",
            "--num_minibatches", "12",
            "--n_train_steps", "20000000",
            "--no-vision",
            "--episode_length", "400",
            "--distance_reach_metric_coefficient", f"{dist_coef:.3f}",
            "--x_reach_weight", f"{x_weight:.3f}",
            "--success_bonus", f"{success_bonus:.1f}",
            "--phase_0_to_1_transition_bonus", "0.0",
            "--policy_hidden_layer_sizes", "128", "128", "128", "128",
            "--value_hidden_layer_sizes", "256", "256", "256", "256", "256",
            "--n_eval_eps", "10",
            "--screen_friction", "1"
        ]
        cmds1.append(cmd1)

    print(f"Starte Batch cmd1 mit {len(cmds1)} Jobs...")
    run_batch(cmds1)
    print("Batch cmd1 fertig.")
