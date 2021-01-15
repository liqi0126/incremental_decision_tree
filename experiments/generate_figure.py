import argparse
import pickle
from matplotlib import pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Incremental Decision Tree: Final Experiments')
parser.add_argument('--dataset', required=True, type=str)
args = parser.parse_args()

def get_info(path):
    vfdt = pickle.load(open(path, 'rb'))
    vfdt_performance = vfdt['performances'][0]
    vfdt_time = vfdt['total_time']
    return vfdt_performance, vfdt_time

"""
Unshuffle
"""
vfdt_pl = f'../outputs/vfdt_{args.dataset}_final_unshuffle.pickle'
efdt_pl = f'../outputs/efdt_{args.dataset}_final_unshuffle.pickle'

vfdt_performance, vfdt_time = get_info(vfdt_pl)
efdt_performance, efdt_time = get_info(efdt_pl)

plt.plot(vfdt_performance, '--', linewidth=1, color='#8DBD85')
plt.plot(efdt_performance, 'black', linewidth=1)

plt.title("%s dataset: unshuffled" % args.dataset)
plt.xlabel("Instances (x 1,000)")
plt.ylabel("Error rate")
plt.legend(labels=[
    "VFDT: | T: %.2f s | E: %.4f" % (vfdt_time, np.mean(vfdt_performance[-10:])),
    "EFDT: | T: %.2f s | E: %.4f" % (efdt_time, np.mean(efdt_performance[-10:]))
])
# plt.show()
plt.savefig(f'{args.dataset}_final_unshuffle.png', dpi=600)
plt.cla()

"""
Shuffle
"""
vfdt_performances, vfdt_times = [], []
efdt_performances, efdt_times = [], []
for seed in range(4096, 4106):
    vfdt_pl = f'../outputs/vfdt_{args.dataset}_final_shuffle_seed{seed}.pickle'
    efdt_pl = f'../outputs/efdt_{args.dataset}_final_shuffle_seed{seed}.pickle'

    vfdt_performance, vfdt_time = get_info(vfdt_pl)
    efdt_performance, efdt_time = get_info(efdt_pl)

    vfdt_performances.append(vfdt_performance)
    efdt_performances.append(efdt_performance)
    vfdt_times.append(vfdt_time)
    efdt_times.append(efdt_time)

plt.plot(np.array(vfdt_performances).mean(0), '--', linewidth=1, color='#8DBD85')
plt.plot(np.array(efdt_performances).mean(0), 'black', linewidth=1)

plt.title("%s dataset: 10 stream shuffled average" % args.dataset)
plt.xlabel("Instances (x 1,000)")
plt.ylabel("Error rate")
plt.legend(labels=[
    "VFDT: | T: %.2f s | E: %.4f" % (np.mean(vfdt_times), np.array(vfdt_performances).mean(0)[-10:].mean()),
    "EFDT: | T: %.2f s | E: %.4f" % (np.mean(efdt_times), np.array(efdt_performances).mean(0)[-10:].mean())
])
# plt.show()
plt.savefig(f'{args.dataset}_final_10_shuffled_average.png', dpi=600)
plt.cla()