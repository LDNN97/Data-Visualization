# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


# %%
data_o1 = pd.read_csv("MC_Result_Original/EXP 0 log.txt", delim_whitespace=True,  header=None)
data_o2 = pd.read_csv("MC_Result_Original/EXP 1 log.txt", delim_whitespace=True,  header=None)
data_o3 = pd.read_csv("MC_Result_Original/EXP 2 log.txt", delim_whitespace=True,  header=None)
data_o4 = pd.read_csv("MC_Result_Original/EXP 3 log.txt", delim_whitespace=True,  header=None)
data_o5 = pd.read_csv("MC_Result_Original/EXP 4 log.txt", delim_whitespace=True,  header=None)

# %%
line_ol = (data_o1[:][9] + data_o2[:][9] + data_o3[:][9] + data_o4[:][9] + data_o5[:][9]) / 5
line_ot = (data_o1[:][11] + data_o2[:][11] + data_o3[:][11] + data_o4[:][11] + data_o5[:][11]) / 5
line_ol.name = 'original'
line_ot.name = 'original'

# %%
data_i1 = pd.read_csv("MC_Result_Improved/EXP 0 log.txt", delim_whitespace=True,  header=None)
data_i2 = pd.read_csv("MC_Result_Improved/EXP 1 log.txt", delim_whitespace=True,  header=None)
data_i3 = pd.read_csv("MC_Result_Improved/EXP 2 log.txt", delim_whitespace=True,  header=None)
data_i4 = pd.read_csv("MC_Result_Improved/EXP 3 log.txt", delim_whitespace=True,  header=None)
data_i5 = pd.read_csv("MC_Result_Improved/EXP 4 log.txt", delim_whitespace=True,  header=None)

# %%
line_il = (data_i1[:][9] + data_i2[:][9] + data_i3[:][9] + data_i4[:][9] + data_i5[:][9]) / 5
line_it = (data_i1[:][11] + data_i2[:][11] + data_i3[:][11] + data_i4[:][11] + data_i5[:][11]) / 5
line_il.name = 'improved'
line_it.name = 'improved'

# %%
x = range(0, 300)
data_g1 = pd.DataFrame([line_ol, line_il]).T
data_g2 = pd.DataFrame([line_ol, line_it]).T

# %%
fig = plt.figure(figsize=(15, 6), dpi=128)
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}

ax1 = fig.add_subplot(121)
ax1.plot(x, line_ol, '--', label='original', linewidth=3.5)
ax1.plot(x, line_il, '-',  label='improved', linewidth=3.5)
ax1.set_ylim(-200, -80)
ax1.set_xlabel('Generation', font1)
ax1.set_ylabel('Average Reward', font1)
ax1.tick_params(labelsize=15)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# ax1.xaxis.set_major_locator(MultipleLocator(25))
# ax1.yaxis.set_major_locator(MultipleLocator(10))
ax1.grid(True)
ax1.legend(prop=font1)

ax2 = fig.add_subplot(122)
ax2.plot(x, line_ot, '--', label='original', linewidth=3.5)
ax2.plot(x, line_it, '-',label='improved', linewidth=3.5)
ax2.set_ylim(-200, -80)
ax2.set_xlabel('Generation', font1)
ax2.set_ylabel('Average Reward', font1)
ax2.tick_params(labelsize=15)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
# ax2.xaxis.set_major_locator(MultipleLocator(25))
# ax2.yaxis.set_major_locator(MultipleLocator(10))
ax2.grid(True)
ax2.legend(prop=font1)

plt.show()

# %%
bin = range(-200, -80, 5)
dis_o = np.concatenate((data_o1[:][5], data_o2[:][5], data_o3[:][5], data_o4[:][5], data_o5[:][5]), axis=0)
dis_i = np.concatenate((data_i1[:][5], data_i2[:][5], data_i3[:][5], data_i4[:][5], data_i5[:][5]), axis=0)

# _data2 = pd.Series(data=tmp)
# _data2.columns = ["best_individual"]
# _data2.shape
# bins = pd.cut(_data2, bin)
# _data2.groupby(bins).agg('count')

fig2 = plt.figure(figsize=(15, 10), dpi=128)
font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30}
font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 25}

ax3 = fig2.add_subplot(111)
ax3.hist(dis_o, bins=bin, label='original', alpha=0.5)
ax3.hist(dis_i, bins=bin, label='improved', alpha=0.5)
ax3.set_xlabel('fitness', font1)
ax3.set_ylabel('count', font1)
ax3.grid(True)
ax3.xaxis.set_major_locator(MultipleLocator(10))
ax3.yaxis.set_major_locator(MultipleLocator(40))
ax3.tick_params(labelsize=20)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ax3.legend(prop=font1)

plt.show()
