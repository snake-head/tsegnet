import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import interpolate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
log_dir = '2022-01-07_00-21'
log_path = os.path.join(ROOT_DIR, 'log', 'seg', log_dir, 'logs', 'tsegnet.txt')

train_loss_list = list()
train_distance_list = list()
eval_loss_list = list()
eval_distance_list = list()
with open(log_path, 'r') as f:
    while True:
        line = f.readline()
        if line == '':
            break
        infos = line.split()
        for info in infos:
            if info == 'Training':
                train_loss = float(infos[-1])
                if infos[-2] == 'loss:':
                    train_loss_list.append(train_loss)
                else:
                    train_distance_list.append(train_loss)
            if info == 'eval':
                eval_loss = float(infos[-1])
                if infos[-2] == 'loss:':
                    eval_loss_list.append(eval_loss)
                else:
                    eval_distance_list.append(eval_loss)

x = np.arange(1, len(train_loss_list) + 1)
y_train_loss = np.array(train_loss_list)
y_eval_loss = np.array(eval_loss_list)
y_train_distance = np.array(train_distance_list)
y_eval_distance = np.array(eval_distance_list)

plt.figure(facecolor='#FFFFFF', figsize=(8, 5))
plt.title('trian loss vs. eval loss',
          fontsize=22, color=(0.4, 0.4, 0.4), loc='center')

color_dict = {
    'y_train_loss': '#EE6363',
    'y_eval_loss': '#FFC125',
    'alipay': '#4F94CD'
}

cl = color_dict
plt.plot(x, y_train_loss, marker='.', ls='-', label='y_train_loss', c=cl['y_train_loss'], linewidth=1.0, ms=6,
         mfc=cl['y_train_loss'], mec=cl['y_train_loss'], mew=3, mfcalt='m')
plt.plot(x, y_eval_loss, marker='.', ls='-', label='y_eval_loss', c=cl['y_eval_loss'], linewidth=1.0, ms=6,
         mfc=cl['y_eval_loss'], mec=cl['y_eval_loss'], mew=3)

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height])
ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
          ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
plt.show()


plt.figure(facecolor='#FFFFFF', figsize=(8, 5))
plt.title('trian acc vs. eval acc',
          fontsize=22, color=(0.4, 0.4, 0.4), loc='center')

color_dict = {
    'y_train_distance': '#EE6363',
    'y_eval_distance': '#FFC125',
    'alipay': '#4F94CD'
}

cl = color_dict
y_train_distance /= 3.5
plt.plot(x, y_train_distance, marker='.', ls='-', label='y_train_distance', c=cl['y_train_distance'], linewidth=1.0, ms=6,
         mfc=cl['y_train_distance'], mec=cl['y_train_distance'], mew=3, mfcalt='m')
y_eval_distance /= 3.5
plt.plot(x, y_eval_distance[1::3], marker='.', ls='-', label='y_eval_distance', c=cl['y_eval_distance'], linewidth=1.0, ms=6,
         mfc=cl['y_eval_distance'], mec=cl['y_eval_distance'], mew=3)

ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height])
ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
          ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
plt.show()