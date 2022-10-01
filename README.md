# Hierarchical Adversarial Inverse Reinforcement Learning for Robotic Manipulation

## How to config the environments:
- on Ubuntu 18.04
- python 3.6
- pytorch 1.6
- tensorboard 2.5
- mujoco_py >= 1.5
- gym == 0.19.0
- matplotlib
- tqdm
- seaborn
- ...

## Experiments with Hopper
- You need to first enter the folder 'HierAIRL_Hopper'.

- To run the code with specific algorithms:

```bash
# Option-GAIL:
python ./run_baselines.py --env_type mujoco --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" --tag option-gail-1k --algo option_gail

# GAIL:
python ./run_baselines.py --env_type mujoco --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" --tag gail-1k --algo gail

# DI-GAIL:
python ./run_baselines.py --env_type mujoco --env_name Hopper-v2 --n_pretrain_epoch 50 --n_demo 1000 --device "cuda:0" --tag d_info_gail-1k --algo DI_gail

# Option-AIRL
python ./run_main.py --env_type mujoco --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" --tag option-airl-1k --algo option_airl

# H-AIRL
python ./run_main.py --env_type mujoco --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" --tag hier-airl-1k --algo hier_airl

# H-GAIL
python ./run_main.py --env_type mujoco --env_name Hopper-v2 --n_demo 1000 --device "cuda:0" --tag hier-gail-1k --algo hier_gail
```
- To run the code with the random seed Y, for which we simply choose 0, 1, or 2, please add '--seed=Y' to the back. **The same below for other tasks.**

- For the hyperparameters, please refer to 'HierAIRL_Hopper/default_config.py'. **The same below for other tasks.**

## Experiments with Walker
- You need to first enter the folder 'HierAIRL_Walker'.

- To run the code with specific algorithms:

```bash
# Option-GAIL:
python ./run_baselines.py --env_type mujoco --env_name Walker2d-v2 --n_demo 5000 --device "cuda:0" --tag option-gail-5k --algo option_gail

# GAIL:
python ./run_baselines.py --env_type mujoco --env_name Walker2d-v2 --n_demo 5000 --device "cuda:0" --tag gail-5k --algo gail

# DI-GAIL:
python ./run_baselines.py --env_type mujoco --env_name Walker2d-v2 --n_pretrain_epoch 50 --n_demo 5000 --device "cuda:0" --tag d_info_gail-5k --algo DI_gail

# Option-AIRL
python ./run_main.py --env_type mujoco --env_name Walker2d-v2 --n_demo 5000 --device "cuda:0" --tag option-airl-5k --algo option_airl

# H-AIRL:
python ./run_main.py --env_type mujoco --env_name Walker2d-v2 --n_demo 5000 --device "cuda:0" --tag hier-airl-5k --algo hier_airl

# H-GAIL:
python ./run_main.py --env_type mujoco --env_name Walker2d-v2 --n_demo 5000 --device "cuda:0" --tag hier-gail-5k --algo hier_gail
```

## Experiments with AntPusher
- You need to first enter the folder 'HierAIRL_Ant'.

- To run the code with specific algorithms:

```bash
# Option-GAIL:
python ./run_baselines.py --env_type mujoco --env_name AntPusher-v0 --n_demo 10000 --device "cuda:0" --tag option-gail-10k --algo option_gail

# GAIL:
python ./run_baselines.py --env_type mujoco --env_name AntPusher-v0 --n_demo 10000 --device "cuda:0" --tag gail-10k --algo gail

# DI-GAIL:
python ./run_baselines.py --env_type mujoco --env_name AntPusher-v0 --n_pretrain_epoch 100 --n_demo 10000 --device "cuda:0" --tag d_info_gail-10k --algo DI_gail

# Option-AIRL:
python ./run_main.py --env_type mujoco --env_name AntPusher-v0 --n_demo 10000 --device "cuda:0" --tag option-airl-10k --algo option_airl

# H-AIRL:
python ./run_main.py --env_type mujoco --env_name AntPusher-v0 --n_demo 10000 --device "cuda:0" --tag hier-airl-10k --algo hier_airl

# H-GAIL:
python ./run_main.py --env_type mujoco --env_name AntPusher-v0 --n_demo 10000 --device "cuda:0" --tag hier-gail-10k --algo hier_gail
```

## Experiments on Point Room/Corridor
- You need to first enter the folder 'HierAIRL_Point'.

- To reproduce the results of expert trajectories, please run the following command, where XXX can be Point4Rooms-v1 or PointCorridor-v1. The results will be available in the folder 'result'.
```bash
python ./plot_options_exp.py --env_type mujoco --env_name XXX
```

- To reproduce the results of trajectories of the learned agents, please run the following command, where XXX can be Point4Rooms-v1 or PointCorridor-v1. The results will be available in the folder 'result'.
```bash
python ./plot_options.py --env_type mujoco --env_name XXX
```

- To reproduce the learned agents with H-AIRL (i.e., the checkpoints), please run the following command, where XXX can be Point4Rooms-v1 or PointCorridor-v1. The results will be available in the folder 'result'.
```bash
python ./run_main.py --env_type mujoco --env_name XXX --n_demo 5000--device "cuda:0" --tag hier-airl-5k --algo hier_airl
```

## Transfer Learning Results on Point Room
- You need to first enter the folder 'HierAIRL_Point_Room_transfer'.

- To run the code with specific algorithms, please run the following commands, where X can be 0, 1, 2.
```bash
# Option-GAIL:
python ./run_baselines.py --env_type mujoco --env_name Point4Rooms-v1 --n_demo 5000 --device "cuda:0" --tag option-gail-5k --algo option_gail --seed X

# GAIL:
python ./run_baselines.py --env_type mujoco --env_name Point4Rooms-v1 --n_demo 5000 --device "cuda:0" --tag gail-5k --algo gail --seed X

# H-AIRL:
python ./run_main.py --env_type mujoco --env_name Point4Rooms-v1 --n_demo 5000 --device "cuda:0" --tag hier-airl-5k --algo hier_airl --seed X --init 0

# H-AIRL initialized with the checkpoint trained in another task:
python ./run_main.py --env_type mujoco --env_name Point4Rooms-v1 --n_demo 5000 --device "cuda:0" --tag hier-airl-5k --algo hier_airl --seed X --init 1
```

## Transfer Learning Results on Point Corridor
- You need to first enter the folder 'HierAIRL_Point_Corridor_transfer'.

- To run the code with specific algorithms, please run the following commands.
```bash
# Option-GAIL:
python ./run_baselines.py --env_type mujoco --env_name PointCorridor-v1 --n_demo 5000 --device "cuda:0" --tag option-gail-5k --algo option_gail --seed X

# GAIL:
python ./run_baselines.py --env_type mujoco --env_name PointCorridor-v1 --n_demo 5000 --device "cuda:0" --tag gail-5k --algo gail --seed X

# H-AIRL:
python ./run_main.py --env_type mujoco --env_name PointCorridor-v1 --n_demo 5000 --device "cuda:0" --tag hier-airl-5k --algo hier_airl --seed X --init 0

# H-AIRL initialized with the checkpoint trained in another task:
python ./run_main.py --env_type mujoco --env_name PointCorridor-v1 --n_demo 5000 --device "cuda:0" --tag hier-airl-5k --algo hier_airl --seed X --init 1
```

