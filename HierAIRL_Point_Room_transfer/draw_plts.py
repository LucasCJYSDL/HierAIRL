import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot():
    sns.set_theme(style="darkgrid")
    
    # data files
    common_dir = './log_saved'
    file_dir = {'H-AIRL': [0, 1, 2], 'H-AIRL-init': [3, 4, 5], 'GAIL': [9, 10, 11], 'Option-GAIL': [6, 7, 8]}


    data_frame = pd.DataFrame()
    for alg, dir_name_list in file_dir.items():
        print("Processing ", alg)
        for dir_name in dir_name_list:
            csv_file_name = str(dir_name) + '.csv'
            csv_file_dir = os.path.join(common_dir, csv_file_name)
            print("Loading from: ", alg, csv_file_dir)

            temp_df = pd.read_csv(csv_file_dir)
            temp_step = np.array(temp_df['Step'])
            temp_value = np.array(temp_df['Value'])
            print("Average rwd across the episodes: ", np.mean(temp_value))
            temp_len = len(temp_step)
            
            mov_max_agent = MovAvg(mode='max')
            for i in range(temp_len):
                cur_value = mov_max_agent.update(temp_value[i])
                temp_value[i] = cur_value

            mov_avg_agent = MovAvg()
            for i in range(temp_len):
                cur_value = mov_avg_agent.update(temp_value[i])
                data_frame = data_frame.append({'algorithm': alg, 'Step': temp_step[i] * 4096, 'Reward': cur_value}, ignore_index=True)

    # expert value: 168.46 for room
    for i in range(temp_len):
        data_frame = data_frame.append({'algorithm': 'Expert', 'Step': temp_step[i] * 4096, 'Reward': 168.46}, ignore_index=True)

    sns.set(font_scale=1.5)
    pal = sns.xkcd_palette((['red', 'blue', 'green', 'orange', 'yellow']))
    g = sns.relplot(x="Step", y="Reward", hue='algorithm', kind="line", ci="sd", data=data_frame, legend='brief', palette=pal)

    leg = g._legend
    leg.set_bbox_to_anchor([0.69, 0.38])
    g.fig.set_size_inches(15, 6)
    plt.savefig(common_dir + '/' + 'Reward.png')


class MovAvg(object):

    def __init__(self, window_size=15, mode='avg'):
        self.window_size = window_size
        self.data_queue = []
        self.mode = mode

    def set_window_size(self, num):
        self.window_size = num

    def clear_queue(self):
        self.data_queue = []

    def update(self, data):
        if len(self.data_queue) == self.window_size:
            del self.data_queue[0]
        self.data_queue.append(data)
        if self.mode == 'avg':
            return sum(self.data_queue) / len(self.data_queue)
        else:
            return max(self.data_queue)


if __name__ == '__main__':
    plot()


