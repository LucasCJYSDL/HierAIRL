import os
from typing import Tuple, List, Dict
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class MazeCell(Enum):
    ROBOT = -1
    EMPTY = 0
    BLOCK = 1

E, B, R = MazeCell.EMPTY, MazeCell.BLOCK, MazeCell.ROBOT
SCALE = 4
MAZE = {
        '4Rooms': [
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, E, B, B, B, B, B, B, B, E, E, B, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
            [B, E, E, E, B, B, B, B, E, E, B, E, B, E, B],
            [B, E, E, E, E, E, E, E, E, E, B, E, B, E, B],
            [B, E, B, E, E, E, E, E, E, E, B, E, B, E, B],
            [B, E, B, E, B, E, E, R, E, E, B, E, B, E, B],
            [B, E, B, E, B, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, E, B, E, E, E, E, E, E, E, E, E, B],
            [B, E, B, E, B, E, E, B, B, B, B, E, E, E, B],
            [B, E, B, E, E, E, E, E, E, E, E, E, B, E, B],
            [B, E, B, B, E, E, B, B, B, B, B, B, B, E, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B, B, B]
        ],

        "Corridor": [
            [B, B, B, B, B, B, B, B, B, B, B, B, B],
            [B, E, E, E, E, B, E, E, E, E, E, E, B],
            [B, E, B, B, E, B, E, B, B, B, B, E, B],
            [B, E, B, B, E, B, E, B, B, B, B, E, B],
            [B, E, B, B, B, B, E, B, B, E, E, R, B],
            [B, E, B, B, B, E, E, E, B, B, B, B, B],
            [B, E, E, E, E, E, E, E, E, E, E, E, B],
            [B, B, B, B, B, E, E, E, B, B, B, E, B],
            [B, E, E, E, B, B, E, B, B, B, B, E, B],
            [B, E, B, B, B, B, E, B, E, B, B, E, B],
            [B, E, B, B, B, B, E, B, E, B, B, E, B],
            [B, E, E, E, E, E, E, B, E, E, E, E, B],
            [B, B, B, B, B, B, B, B, B, B, B, B, B]
        ]
        }

def _get_lower_left_loc(i_idx: int, j_idx: int, origin_idx: Tuple):
    ori_i = origin_idx[0]
    ori_j = origin_idx[1]

    y = (i_idx - ori_i) * SCALE + 0.0 - SCALE / 2.0
    x = (j_idx - ori_j) * SCALE + 0.0 - SCALE / 2.0

    return (x, y)


def draw_traj(env_id: str, option_num: int, trajectory_list: Dict, unique_token: str, episode_id: int):

    if '4Rooms' in env_id:
        maze = MAZE['4Rooms']
        origin_idx = (7, 7)
    else:
        assert 'Corridor' in env_id, env_id
        maze = MAZE['Corridor']
        origin_idx = (4, 11)

    maze_size = len(maze)

    # preparation
    cmap = plt.get_cmap('viridis', option_num)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # # draw the trajectories
    for c_id in range(option_num):
        c_traj_list = trajectory_list[c_id]
        for traj_id in range(len(c_traj_list)):
            ax.scatter(c_traj_list[traj_id]['x'], c_traj_list[traj_id]['y'], color=cmap(c_id), alpha=1.0)
            # ax.plot(c_traj_list[traj_id]['x'][0], c_traj_list[traj_id]['y'][0], marker='o', markersize=5, color='red',
            #         zorder=11, alpha=0.5)
            # ax.plot(c_traj_list[traj_id]['x'][1:], c_traj_list[traj_id]['y'][1:], label="Option #{}".format(c_id),
            #         color=cmap(c_id), alpha=1.0, linewidth=2, zorder=10)
            for i in range(len(c_traj_list[traj_id]['x'])-1):
                if (c_traj_list[traj_id]['x'][i] - c_traj_list[traj_id]['x'][i+1])**2 + (c_traj_list[traj_id]['y'][i] - c_traj_list[traj_id]['y'][i+1])**2 <= 16:
                    ax.plot([c_traj_list[traj_id]['x'][i], c_traj_list[traj_id]['x'][i+1]], [c_traj_list[traj_id]['y'][i], c_traj_list[traj_id]['y'][i+1]], color=cmap(c_id), alpha=1.0, linewidth=2, zorder=10)
    # draw the maze
    for i in range(maze_size):
        for j in range(maze_size):
            if maze[i][j] == B:
                loc = _get_lower_left_loc(i, j, origin_idx)
                ax.add_patch(Rectangle(loc, SCALE, SCALE, edgecolor='gray', facecolor='gray', fill=True, alpha=0.5))

    # eliminate the redundant parts
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    for p in ["left", "right", "top", "bottom"]:
        ax.spines[p].set_visible(False)

    # plt.show()
    result_path = "./result"
    save_path = os.path.abspath(os.path.join(result_path, unique_token))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig_path = os.path.join(save_path, str(episode_id) + '_r' + '.png')
    plt.savefig(fig_path)

if __name__ == '__main__':

    traj_list = {0: [{'x': [0, 1], 'y': [0, 0]}, {'x': [0, 1], 'y': [-1, -1]}],
                 1: [{'x': [0, -1], 'y': [0, 0]}, {'x': [0, -1], 'y': [-1, -1]}]}
    draw_traj('Point4Corridor', 2, traj_list, 'test', 100)