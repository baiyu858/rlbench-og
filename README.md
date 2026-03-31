<p align="center">
    <h1 align="center">
        RLBench-OG Benchmark [CVPR'26]
    </h1>
</p>

<div align="center">
  <p>
    <a href="https://hcplab-sysu.github.io/TAVP/">
      <img src="https://img.shields.io/badge/Website-grey?logo=google-chrome&logoColor=white&labelColor=blue">
    </a>
    <a href="https://arxiv.org/pdf/2508.05186">
      <img src="https://img.shields.io/badge/arXiv-grey?logo=arxiv&logoColor=white&labelColor=red">
    </a>
    <a href="https://huggingface.co/datasets/baiyu858/RLBench-OG">
      <img src="https://img.shields.io/badge/%F0%9F%A4%97-Huggingface-yellow">
    </a>
  </p>
</div>
<br>

## Introduction

RLBench-OG is an extension benchmark built on top of RLBench to evaluate model robustness under occlusion and generalization to environment perturbations. The benchmark selects ten tasks from RLBench (covering simple and long-horizon tasks) and contains two main components: the Occlusion Suite and the Generalization Suite.

## Benchmark Overview

- Occlusion Suite: Occlusions are introduced to the `front_camera` via two mechanisms:
  1. Self-occlusion by perturbing object poses (e.g., rotating a drawer to occlude the handle).
  2. External occluders placed in front of the workspace (cabinets, TVs, doors, etc.).

  Example task-specific occlusion configurations:
  - `basketball_in_hoop`: basket and trash can pose perturbations occlude the basketball
  - `block_pyramid`: a cabinet is placed in front of the workspace to occlude some blocks
  - `close_drawer`: the drawer is rotated so its geometry occludes the handle
  - `scoop_with_spatula`: a wine bottle blocks the target cube
  - `solve_puzzle`: a storage cabinet occludes puzzle pieces
  - `straighten_rope`: a desk lamp occludes one end of the rope
  - `take_plate_off_colored_dish_rack`: a box and laptop block the plate
  - `take_usb_out_of_computer`: a cabinet blocks the USB port area
  - `toilet_seat_down`: a door occludes the toilet seat
  - `water_plants`: a TV partially blocks the watering can and plant

- Generalization Suite: Evaluates robustness to single-factor environment variations. Based on the same ten tasks, each dataset variant changes exactly one factor:
  - `light_color`: sample RGB values for directional lights
  - `table_texture`: sample textures from a texture dataset and apply to the table
  - `table_color`: sample RGB values for the table surface
  - `background_texture`: apply background textures randomly
  - `distractor`: spawn two distractor objects sampled from a 3D asset dataset
  - `camera_pose`: apply position/orientation offsets to front, left-shoulder, and right-shoulder cameras

Each configuration stores multi-view RGB-D images, robot state, and robot actions suitable for imitation learning, reinforcement learning, and evaluation.

## Environment Setup (refer to RLBench)

This dataset is built on RLBench + PyRep + CoppeliaSim. We recommend following the RLBench installation instructions. Below are the key steps and common commands (Ubuntu examples). For full details see the RLBench repository: https://github.com/stepjam/RLBench

1) Install CoppeliaSim (example):

```bash
# Set environment variables (adjust paths as needed)
export COPPELIASIM_ROOT=${HOME}/CoppeliaSim
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# Download and extract (example: CoppeliaSim v4.1.0)
wget https://downloads.coppeliarobotics.com/V4_1_0/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
mkdir -p $COPPELIASIM_ROOT && tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz -C $COPPELIASIM_ROOT --strip-components 1
rm -f CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
```

2) Install RLBench Python package and dependencies (example):

```bash
# Install RLBench (this will pull PyRep and related dependencies; check RLBench docs for platform-specific notes)
pip install git+https://github.com/stepjam/RLBench.git

# Or install from this repo's requirements file if present
pip install -r requirements.txt
```

3) Headless servers: If you run on a headless machine, follow the RLBench headless guide to configure an X server (e.g., create a virtual display `:99`) or use a container with GPU passthrough. The RLBench README contains complete headless instructions.

Note: The data collection scripts in this repository assume RLBench / PyRep / CoppeliaSim are installed and accessible from Python.

## Data collection scripts

This repository provides two main utilities for dataset generation: `collect_dataset.sh` and `batch_collect_data.py`.

- collect_dataset.sh
  - Purpose: collect Generalization Suite data (per-task / per-variation demonstrations).
  - Usage: run the script directly:

    ```bash
    ./collect_dataset.sh
    ```

    To change which tasks or which modalities / image sizes to collect, edit the variables inside the `collect_dataset.sh` file before running.

- batch_collect_data.py
  - Purpose: batch-collect data for the Occlusion Suite and the corresponding normal (non-occlusion) sets. It runs tasks in the sequence: occlusion train (50 eps), occlusion test (25 eps), then normal train (50 eps).
  - Usage: start it with Python:

    ```bash
    python batch_collect_data.py
    ```

    Adjust collection paths, episode counts, parallelism and task list by editing the top-level configuration variables in `batch_collect_data.py`.

Both scripts assume RLBench / PyRep / CoppeliaSim are installed and accessible from your Python environment. The scripts perform basic checks (e.g. counting existing `episode*` folders) and support resuming by skipping already-completed tasks.

## Data structure & visualization

Data are organized by task and variation. Each task/variation contains an `episodes` directory with per-episode folders. Each episode includes multi-view RGB / Depth (if enabled), robot state, and action logs. Use `tools/visualize_task.py` or repository `resources` for quick visualization of collected episodes.

## Visualizations

For visualizations of different variant settings corresponding to each task, refer to the figures below:

![Variant Visualizations Part 1](Fig/Fig-og-1.png)
*Visualization of different variants for the **basketball_in_hoop**, **block_pyramid**, **close_drawer**, **scoop_with_spatula**, **solve_puzzle** tasks.*

![Variant Visualizations Part 2](Fig/Fig-og-2.png)
*Visualization of different variants for the **straighten_rope**, **take_plate_off_colored_dish_rack**, **take_usb_out_of_computer**, **toilet_seat_down**, **water_plants** tasks.*

## Acknowledgement

We extend our sincere gratitude to the following open-source projects: [PyRep](https://github.com/stepjam/PyRep.git), [RLBench](https://github.com/stepjam/RLBench), and [Colosseum](https://github.com/robot-colosseum/robot-colosseum.git). Our RLBench-OG Benchmark is built upon their foundations.

## Citation & License

If you use this dataset in your research, please cite:

```
@InProceedings{Bai_2026_CVPR,
  author    = {Bai, Yongjie and Wang, Zhouxia and Liu, Yang and Luo, Kaijun and Wen, Yifan and Dai, Mingtong and Chen, Weixing and Chen, Ziliang and Liu, Lingbo and Li, Guanbin and Lin, Liang},
  title     = {Learning to See and Act: Task-Aware Virtual View Exploration for Robotic Manipulation},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2026}
}
```

This dataset is released under the MIT License.

## Contact

For questions about this dataset, contact: baiyj26@mail2.sysu.edu.cn

---

(See the `tools/` and `colosseum/` subdirectories for more details, example configs, and collection parameters.)
