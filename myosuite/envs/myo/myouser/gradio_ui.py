import gradio as gr
if gr.NO_RELOAD:
    from gradio_rangeslider import RangeSlider
    from myosuite.envs.myo.myouser.myouser_universal import MyoUserUniversal
    from myosuite.envs.myo.myouser.train_jax_ppo import train
    from myosuite.envs.myo.myouser.hydra_cli import load_config_interactive
    import numpy as np
    import myosuite
    import os
    from pathlib import Path
    myosuite_path = Path(myosuite.__path__[0])


sphere_ranges = {
    "x": (0.225, 0.35),
    "y": (-0.1, 0.1),
    "z": (-0.3, 0.3),
    "size": (0.05, 0.15),
}

INIT_ELEMENTS = 2


parent_path = myosuite_path.parent
CHECKPOINT_PATH = os.path.join(parent_path, "tracked_checkpoints/universal")


def get_available_checkpoints():
    checkpoints = os.listdir(CHECKPOINT_PATH)
    checkpoints.append("None")
    return checkpoints


def is_number(s: str):
    if ("." not in s) and s.isdigit():
        return True
    return False
    
def get_available_checkpoint_numbers(checkpoint_run):
    if checkpoint_run == "None":
        return ["None"]
    checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_run)
    checkpoint_path = os.path.join(checkpoint_path, "checkpoints")
    available_numbers = [num for num in os.listdir(checkpoint_path) if is_number(num)]
    return available_numbers


def checkpoint_path_from_run_number(checkpoint_run, checkpoint_number):
    if checkpoint_run == "None" or checkpoint_number == "None":
        return "None"
    checkpoint_path = os.path.join(CHECKPOINT_PATH, checkpoint_run)
    checkpoint_path = os.path.join(checkpoint_path, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_path, checkpoint_number)
    return checkpoint_path

def update_checkpoint_numbers(checkpoint_run):
    choices = get_available_checkpoint_numbers(checkpoint_run)
    return gr.update(choices=choices, value=choices[0])

def extract_rgb(rgba):
    rgba = rgba.split("rgba(")[1].strip(")")
    r, g, b, a = rgba.split(",")
    r = r.strip()
    g = g.strip()
    b = b.strip()
    rgb = [float(x.strip()) / 255.0 for x in [r, g, b]]
    return rgb

def hex_to_rgb(hex_color):
    if "rgba" in hex_color:
        return extract_rgb(hex_color)
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
    return rgb


def get_ui(wandb_url, save_cfgs=[]):
    # Fix box handlers properly
    with gr.Blocks() as demo:
        gr.Markdown("**Weights & Biases URL:**")
        url_display = gr.Textbox(
            value=wandb_url,
            label="Results Dashboard",
            interactive=False,
            show_copy_button=True,
            info="Click to copy the URL and monitor training progress"
                    )
        gr.Markdown("### RL Parameters")
        with gr.Row():
            num_timesteps = gr.Number(
                label="Number of Timesteps",
                value=15000000,
                minimum=0,
                maximum=50000000,
                interactive=True
            )
            num_checkpoints = gr.Number(
                label="Number of Checkpoints",
                value=1,
                minimum=1,
                maximum=10,
                interactive=True
            )
            num_evaluations = gr.Number(
                label="Number of Evaluations during Training",
                value=1,
                minimum=1,
                maximum=10,
                interactive=True
            )
            batch_size = gr.Number(
                label="Batch Size",
                value=128,
                minimum=0,
                maximum=512,
                interactive=True
            )
            num_envs = gr.Number(
                label="Number of Environments",
                value=1024,
                minimum=0,
                maximum=4096,
                interactive=True
            )
            num_minibatches = gr.Number(
                label="Number of Minibatches",
                value=8,
                minimum=0,
                maximum=40,
                interactive=True
            )
        gr.Markdown(
            "<span style='font-size: 1em;'>"
            "Note: Ensure that batch_size * num_minibatches % num_envs = 0."
            "</span>",
            elem_id="hint-text"
        )
        with gr.Row():
            choices = get_available_checkpoints()
            select_checkpoint_run = gr.Dropdown(
                label="Select Experiment from which to load checkpoints",
                choices=choices,
                interactive=True,
                value="None"
            )
            choices = get_available_checkpoint_numbers(select_checkpoint_run.value)
            select_checkpoint_number = gr.Dropdown(
                label="Select Checkpoint Number",
                choices=choices,
                interactive=True,
                value="None"
            )
            select_checkpoint_run.change(
                update_checkpoint_numbers,
                inputs=select_checkpoint_run,
                outputs=select_checkpoint_number
            )
        gr.Markdown("### Dynamic Targets Feature")
        
        num_elements = gr.Number(
            label="Number of Targets",
            value=INIT_ELEMENTS,
            minimum=0,
            maximum=10,
            precision=0,
            interactive=True
        )
        
        # Create 10 sets of elements (maximum possible), first INIT_ELEMENTS visible by default
        dynamic_rows = []
        radios = []
        box_rows = []
        sphere_rows = []
        
        # Store all components for easy access
        all_components = {
            'boxes': [],
            'spheres': []
        }
        
        for i in range(10):
            # Main row with radio selection
            with gr.Row(visible=(i < INIT_ELEMENTS)) as main_row:
                with gr.Column():
                    radio = gr.Radio(
                        choices=["Box", "Sphere"],
                        label=f"Target {i+1} Type",
                        value="Sphere",
                        interactive=True
                    )
                    
                    # Boxes option row
                    with gr.Row(visible=False) as box_row:
                        with gr.Accordion(label=f"Box {i+1} Settings", open=False):
                            with gr.Row():
                                box_position_x = gr.Slider(
                                    label="Depth Position",
                                    minimum=0.2,
                                    maximum=0.55,
                                    value=0.225,
                                    step=0.001,
                                    interactive=True
                                )
                                box_position_y = gr.Slider(
                                    label="Horizontal Position",
                                    minimum=-0.25,
                                    maximum=0.25,
                                    value=-0.1,
                                    step=0.001,
                                    interactive=True
                                )
                                box_position_z = gr.Slider(
                                    label="Vertical Position",
                                    minimum=0.6,
                                    maximum=1.2,
                                    value=0.843,
                                    step=0.001,
                                    interactive=True
                                )
                            with gr.Row():
                                box_size_x_slider = gr.Slider(
                                    label="Width",
                                    minimum=0.02,
                                    maximum=0.03,
                                    value=0.025,
                                    step=0.001,
                                    interactive=True
                                )
                                box_size_y_slider = gr.Slider(
                                    label="Height",
                                    minimum=0.02,
                                    maximum=0.03,
                                    value=0.025,
                                    step=0.001,
                                    interactive=True
                                )
                                completion_bonus_btn = gr.Slider(
                                    label="Completion Bonus",
                                    minimum=0.0,
                                    maximum=10.0,
                                    value=0.0,
                                    step=0.1,
                                    interactive=True
                                )
                            with gr.Row():
                                min_touch_force = gr.Slider(
                                    label="Minimum Touch Force",
                                    info="Minimum force required to register a touch",
                                    minimum=0.0,
                                    maximum=10.0,
                                    value=1.0,
                                    step=0.1,
                                    interactive=True
                                )
                                orientation_angle = gr.Slider(
                                    label="Orientation Angle",
                                    minimum=0.0,
                                    maximum=180.0,
                                    value=45,
                                    step=1.0,
                                    interactive=True
                                )
                                rgb_btn = gr.ColorPicker(
                                    label="RGB",
                                    value="#FF6B6B",
                                    interactive=True
                                )
                    
                    # Store box components
                    all_components['boxes'].append({
                        'box_position_x': box_position_x,
                        'box_position_y': box_position_y,
                        'box_position_z': box_position_z,
                        'box_size_x': box_size_x_slider,
                        'box_size_y': box_size_y_slider,
                        'completion_bonus': completion_bonus_btn,
                        'min_touch_force': min_touch_force,
                        'orientation_angle': orientation_angle,
                        'rgb': rgb_btn
                    })
                                
                    # Sphere option row  
                    with gr.Row(visible=True) as sphere_row:
                        with gr.Accordion(label=f"Sphere {i+1} Settings", open=False):
                            with gr.Row():
                                gr.Markdown("#### The coordinates for the sphere targets are randomly sampled from a range, please choose them below")
                            with gr.Row():
                                x_slider = RangeSlider(
                                    label=f"Depth Range",
                                    minimum=sphere_ranges["x"][0],
                                    maximum=sphere_ranges["x"][1],
                                    value=(sphere_ranges["x"][0], sphere_ranges["x"][1]),
                                    step=0.001,
                                    interactive=True
                                )
                                y_slider = RangeSlider(
                                    label=f"Horizontal Range",
                                    minimum=sphere_ranges["y"][0],
                                    maximum=sphere_ranges["y"][1],
                                    value=(sphere_ranges["y"][0], sphere_ranges["y"][1]),
                                    step=0.001,
                                    interactive=True
                                )
                                z_slider = RangeSlider(
                                    label=f"Vertical Range",
                                    minimum=sphere_ranges["z"][0],
                                    maximum=sphere_ranges["z"][1],
                                    value=(sphere_ranges["z"][0], sphere_ranges["z"][1]),
                                    step=0.001,
                                    interactive=True
                                )
                            with gr.Row():
                                size_slider = RangeSlider(
                                    label=f"Size Range",
                                    minimum=sphere_ranges["size"][0],
                                    maximum=sphere_ranges["size"][1],
                                    value=(sphere_ranges["size"][0], sphere_ranges["size"][1]),
                                    step=0.001,
                                    interactive=True
                                )
                                dwell_duration = gr.Number(
                                    label=f"Dwell Duration",
                                    value=0.25,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    interactive=True
                                )
                                completion_bonus_pt = gr.Number(
                                    label=f"Completion Bonus",
                                    value=0.0,
                                    minimum=0.0,
                                    maximum=10.0,
                                    step=0.1,
                                    interactive=True
                                )
                                color_picker = gr.ColorPicker(
                                    label=f"RGB",
                                    value="#FF6B6B",
                                    interactive=True
                                )
                    
                    # Store sphere components
                    all_components['spheres'].append({
                        'x_range': x_slider,
                        'y_range': y_slider,
                        'z_range': z_slider,
                        'size_range': size_slider,
                        'dwell_duration': dwell_duration,
                        'completion_bonus': completion_bonus_pt,
                        'rgb': color_picker
                    })
            
            # Store references
            dynamic_rows.append(main_row)
            radios.append(radio)
            box_rows.append(box_row)
            sphere_rows.append(sphere_row)

        gr.Markdown("### View of the environment")
        render_button = gr.Button("Render Environment", variant="primary", size="lg")
        with gr.Row(visible=False) as env_view_row:
            env_view_1 = gr.Image(label="Environment View", interactive=False)
            env_view_2 = gr.Image(label="Environment View", interactive=False)
        
        # Add Run button and output
        gr.Markdown("### Run Configuration")
        with gr.Row():
            run_button = gr.Button("Run", variant="primary", size="lg")
        
        output_text = gr.Textbox(
            label="Configuration Output",
            lines=20,
            max_lines=30,
            interactive=False,
            show_copy_button=True
        )

        def args_to_cfg_overrides(*args):
            """Print all configuration details"""
            # Extract values from args
            timesteps, checkpoints, evaluations, batch, envs, num_minibatches, num_targets, select_checkpoint_run, select_checkpoint_number = args[:9]
            radio_values = args[9:19]  # 10 radio values

            # Get all other component values
            box_values = args[19:109]  # 9 components × 10 = 90 values
            sphere_values = args[109:]   # 7 components × 10 = 70 values
            

            cfg_overrides = ["env=universal", "run.using_gradio=True", "wandb.project=workshop"]
            to_text = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
            cfg_overrides.append(f"env/task_config/targets={to_text[int(num_targets)]}")

            
            # RL Parameters
            cfg_overrides.append(f"rl.num_timesteps={int(timesteps)}")
            cfg_overrides.append(f"rl.num_checkpoints={int(checkpoints)}")
            cfg_overrides.append(f"rl.num_evals={int(evaluations)}")
            cfg_overrides.append(f"rl.batch_size={int(batch)}")
            cfg_overrides.append(f"rl.num_envs={int(envs)}")
            cfg_overrides.append(f"rl.num_minibatches={int(num_minibatches)}")

            exact_checkpoint_path = checkpoint_path_from_run_number(select_checkpoint_run, select_checkpoint_number)
            cfg_overrides.append(f"rl.load_checkpoint_path={exact_checkpoint_path}")
            error = False

            for i in range(int(num_targets)):
                target_type = radio_values[i]
                cfg_overrides.append(f"+env/task_config/targets/target_{i}={target_type.lower()}")
                
                if target_type == "Box":
                    # Extract box values for this target
                    start_idx = i * 9
                    btn_pos_x = box_values[start_idx]
                    btn_pos_y = -box_values[start_idx + 1]
                    btn_pos_z = box_values[start_idx + 2]
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.position=[{btn_pos_x},{btn_pos_y},{btn_pos_z}]")
                    btn_size_x = box_values[start_idx + 3]
                    btn_size_y = box_values[start_idx + 4]
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.geom_size=[{btn_size_x},{btn_size_y},0.01]")
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.site_size=[{btn_size_x-0.005},{btn_size_y-0.005},0.01]")
                    completion_bonus = box_values[start_idx + 5]
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.completion_bonus={completion_bonus}")
                    min_force = box_values[start_idx + 6]
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.min_touch_force={min_force}")
                    orientation = box_values[start_idx + 7]
                    euler = -orientation * np.pi / 180
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.euler=[0,{euler},0]")
                    rgb = box_values[start_idx + 8]
                    rgb = hex_to_rgb(rgb)
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]")
                    
                    
                else:  # Sphere
                    # Extract sphere values for this target
                    start_idx = i * 7
                    x_range = sphere_values[start_idx]
                    y_range = sphere_values[start_idx + 1]
                    z_range = sphere_values[start_idx + 2]
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.position=[[{x_range[0]},{-y_range[0]},{z_range[0]}],[{x_range[1]},{-y_range[1]},{z_range[1]}]]")
                    size_range = sphere_values[start_idx + 3]
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.size=[{size_range[0]},{size_range[1]}]")
                    dwell = sphere_values[start_idx + 4]
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.dwell_duration={dwell}")
                    if dwell < 0.01:
                        error = True
                        gr.Warning("Dwell duration must be greater than 0.0")
                    completion_bonus = sphere_values[start_idx + 5]
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.completion_bonus={completion_bonus}")
                    rgb = sphere_values[start_idx + 6]
                    rgb = hex_to_rgb(rgb)
                    cfg_overrides.append(f"env.task_config.targets.target_{i}.rgb=[{rgb[0]},{rgb[1]},{rgb[2]}]")

            if error:
                return ["Error in the configuration, please check the configuration and try again"]
            return cfg_overrides

        def run_training(*args):
            cfg_overrides = args_to_cfg_overrides(*args)
            # cfg = load_config_interactive(cfg_overrides, cfg_only=True)
            gr.Info("Set up training start from the GR UI!")
            save_cfgs = []
            save_cfgs.extend(cfg_overrides)
            # train(cfg)
            text = "Go to the next notebook in the cell and run the training!\n"
            text += "\n".join(save_cfgs)
            return text

        def render_environment(*args):
            cfg_overrides = args_to_cfg_overrides(*args)
            print(cfg_overrides)
            config = load_config_interactive(cfg_overrides)
            env = MyoUserUniversal(config.env)
            imgs = env.get_renderings()
            img1 = imgs[0][1]
            img2 = imgs[1][1]
            return gr.update(visible=True), gr.update(value=img1), gr.update(value=img2)
            


        def update_dynamic_elements(num):
            """Show/hide dynamic rows based on the number input"""
            num = max(0, min(int(num) if num is not None else 0, 10))
            updates = []
            for i in range(10):
                if i < num:
                    updates.append(gr.update(visible=True))
                else:
                    updates.append(gr.update(visible=False))
            return updates

        def toggle_interface_type(radio_value):
            """Show appropriate interface based on radio selection"""
            if radio_value == "Box":
                return gr.update(visible=True), gr.update(visible=False)
            else:  # Sphere
                return gr.update(visible=False), gr.update(visible=True)

        # Event handler for dynamic elements
        num_elements.change(
            update_dynamic_elements,
            inputs=num_elements,
            outputs=dynamic_rows
        )

        # Event handlers for each radio button to control interface type
        for i in range(10):
            radios[i].change(
                toggle_interface_type,
                inputs=radios[i],
                outputs=[box_rows[i], sphere_rows[i]]
            )

        # Prepare inputs for run button
        run_inputs = [num_timesteps, num_checkpoints, num_evaluations, batch_size, num_envs, num_minibatches, num_elements, select_checkpoint_run, select_checkpoint_number]
        run_inputs.extend(radios)
        
        # Add all box components
        for i in range(10):
            for key in ['box_position_x', 'box_position_y', 'box_position_z', 'box_size_x', 'box_size_y', 'completion_bonus', 'min_touch_force', 'orientation_angle', 'rgb']:
                run_inputs.append(all_components['boxes'][i][key])
        
        # Add all sphere components
        for i in range(10):
            for key in ['x_range', 'y_range', 'z_range', 'size_range', 'dwell_duration', 'completion_bonus', 'rgb']:
                run_inputs.append(all_components['spheres'][i][key])

        # Run button event
        run_button.click(
            run_training,
            inputs=run_inputs,
            outputs=output_text
        )

        render_button.click(
            render_environment,
            inputs=run_inputs,
            outputs=[env_view_row, env_view_1, env_view_2]
        )

    return demo

if __name__ == "__main__":
    wandb_url = None
    demo = get_ui(wandb_url)
    demo.launch(share=True)