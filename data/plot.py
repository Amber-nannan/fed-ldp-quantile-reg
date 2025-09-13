import re
from matplotlib import pyplot as plt

tau_list = [0.25,0.5,0.75]
r_list = [0.1,0.5,0.9]

def read_log(filepath):
    with open(filepath, 'r') as f:
        lines = f.read()
    return lines

def parse_log(log):
    pattern = r"'mse': ([\d\.]+),"
    mse_str = re.findall(pattern, log)
    mse_list = [float(x) for x in mse_str]
    return mse_list

def plot_mse_trajectory(mse_list):
    plt.plot(mse_list)
    plt.xlabel('round')
    plt.ylabel('mse')
    plt.show()


for tau in tau_list:
    for r in r_list:
        file_path = f'results/tau_{tau}_r_{r}_em_5.log'
        log = read_log(file_path)
        mse_list = parse_log(log)
        plot_mse_trajectory(mse_list)
        plt.title(f'tau={tau}, r={r}')
        plt.savefig(f'figures/tau_{tau}_r_{r}.png')
