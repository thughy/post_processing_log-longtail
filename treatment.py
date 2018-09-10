import os 
import xlwt
import xlrd 
from xlutils.copy import copy 
import numpy as np
import matplotlib.pyplot as plt

base = [0.511078347,0.522873364,0.52127225,0.547067406,0.50141295]
if_g_train_first = 1.0
g_step = 2
d_step = 1

train_step = [i/100.0 for i in range(5000)]
g_train_step = []
for i, step in enumerate(train_step):
	if if_g_train_first == 1.0:
		if step%(g_step+d_step) >= g_step:
			g_train_step.append(0.0)
		else:
			g_train_step.append(1.0)
	else:
		if step%(g_step+d_step) >= d_step:
			g_train_step.append(1.0)
		else:
			g_train_step.append(0.0)

# fig = plt.figure() 
# plt.plot(train_step, g_train_step)

# parameters
current_dir_path = os.getcwd()
case = current_dir_path[3:].split('\\')[-1]
print('this case is ', case)
File_path = 'D:/work/models/IRGAN/results/'

keyword_to_read = 'f_rank'
exl_file_name = keyword_to_read + '.xls'
exl_file_name_acc = 'acc.xls'

d_ckpt_num = []
d_err1 = []
d_err3 = []
d_err5 = []
d_dcg5 = []
d_goodpos = []

g_ckpt_num = []
g_err1 = []
g_err3 = []
g_err5 = []
g_dcg5 = []
g_goodpos = []

list_name = os.listdir()


def exact_info(focus):
	# res[0~4] is err1, err3, err5, dcg5, goodpos
	res = []
	for line in focus:
		info = line.strip().split()
		res.append(info[-1])
	return res

def read_file(file_name, f, keyword):
	print('start reading file ' + file_name)
	i = 0
	lines = f.readlines()
	for line in lines: 
		if keyword in line:
			focus = lines[i+1: i+6]
			res = exact_info(focus)
			break
		i += 1
	print('finish reading file ' + file_name)
	return res

# ##########################################################
g_ckpt_num.append(int(0))
g_err1.append(float(base[0]))
g_err3.append(float(base[1]))
g_err5.append(float(base[2]))
g_dcg5.append(float(base[3]))
g_goodpos.append(float(base[4]))

# start to read ckpt
for i in range(500):
	file_name = 'g_ckpt_'
	file_name += str(i)
	file_name += '.longtail.compare'
	if file_name in list_name:
		with open(file_name, "r") as f:
			res = read_file(file_name, f, keyword_to_read)
			g_ckpt_num.append(int(i))
			g_err1.append(float(res[0]))
			g_err3.append(float(res[1]))
			g_err5.append(float(res[2]))
			g_dcg5.append(float(res[3]))
			g_goodpos.append(float(res[4]))

d_ckpt_num.append(int(0))
d_err1.append(float(base[0]))
d_err3.append(float(base[1]))
d_err5.append(float(base[2]))
d_dcg5.append(float(base[3]))
d_goodpos.append(float(base[4]))
# start to read ckpt
for i in range(500):
	file_name = 'd_ckpt_'
	file_name += str(i)
	file_name += '.longtail.compare'
	if file_name in list_name:
		with open(file_name, "r") as f:
			res = read_file(file_name, f, keyword_to_read)
			d_ckpt_num.append(int(i))
			d_err1.append(float(res[0]))
			d_err3.append(float(res[1]))
			d_err5.append(float(res[2]))
			d_dcg5.append(float(res[3]))
			d_goodpos.append(float(res[4]))

# start to write ckpt
# write to txt file
txtName = keyword_to_read
txtName += "_res.txt"

temp = ''
base_err1 = []
base_err3 = []
base_err5 = []
base_dcg5 = []
with open(txtName, "w")  as f:
	for i in range(max([len(d_ckpt_num), len(g_ckpt_num)])):
		base_err1.append(float(base[0]))
		base_err3.append(float(base[1]))
		base_err5.append(float(base[2]))
		base_dcg5.append(float(base[3]))

		if i < len(d_ckpt_num) and i < len(g_ckpt_num):
			temp += str(d_ckpt_num[i]) + '  ' + str(d_err1[i])    + '  ' + \
					str(d_err3[i])     + '  ' + str(d_err5[i])    + '  ' + \
					str(d_dcg5[i])     + '  ' + str(d_goodpos[i]) + '  ' + \
					'//'               + '  ' + str(g_err1[i])    + '  ' + \
					str(g_err3[i])     + '  ' + str(g_err5[i])    + '  ' + \
					str(g_dcg5[i])     + '  ' + str(g_goodpos[i]) + '\n'

		elif i >= len(d_ckpt_num):
			temp += str(g_ckpt_num[i]) + '  ' + 'null'            + '  ' + \
					'null'             + '  ' + 'null'            + '  ' + \
					'null'             + '  ' + 'null'            + '  ' + \
					'//'               + '  ' + str(g_err1[i])    + '  ' + \
					str(g_err3[i])     + '  ' + str(g_err5[i])    + '  ' + \
					str(g_dcg5[i])     + '  ' + str(g_goodpos[i]) + '\n'
		else:
			temp += str(d_ckpt_num[i]) + '  ' + str(d_err1[i])    + '  ' + \
					str(d_err3[i])     + '  ' + str(d_err5[i])    + '  ' + \
					str(d_dcg5[i])     + '  ' + str(d_goodpos[i]) + '  ' + \
					'//'               + '  ' + 'null'            + '  ' + \
					'null'             + '  ' + 'null'            + '  ' + \
					'null'             + '  ' + 'null'            + '\n'
	f.write(temp)

def make_upper_lower(g_train_step, l_max, l_min):
	g_train_fill_upper = [a*l_max for a in g_train_step]
	g_train_fill_lower = [l_min for i in range(len(g_train_fill_upper))]
	for i,a in enumerate(g_train_fill_upper):
		if a < l_min:
			g_train_fill_upper[i] = l_min
	return g_train_fill_upper, g_train_fill_lower

alpha = 0.25
fig = plt.figure() 
ax1 = fig.add_subplot(221)
ax1.plot(g_ckpt_num, g_err1, color = "g", marker='o', label="g_longtail")
ax1.plot(d_ckpt_num, d_err1, color = "b", marker='o', label="d_longtail")
ax1.scatter(d_ckpt_num, base_err1, color = "k", label="base")
l_min = min([min(g_err1), min(d_err1)])*0.999
l_max = max([max(g_err1), max(d_err1)])*1.001
# ax1.set_yticks(np.linspace(l_min, l_max, 5))
ax1.set_ylim(l_min, l_max)
ax1.set_xlim(0, max(d_ckpt_num))
ax1.set_title('err1')
g_train_fill_upper, g_train_fill_lower = make_upper_lower(g_train_step, l_max, l_min)
ax1.fill_between(train_step, g_train_fill_lower, g_train_fill_upper, facecolor='red', alpha=alpha)
ax1.legend(loc='upper left')

ax2 = fig.add_subplot(222)
ax2.plot(g_ckpt_num, g_err3, color = "g", marker='o', label="g_longtail")
ax2.plot(d_ckpt_num, d_err3, color = "b", marker='o', label="d_longtail")
ax2.scatter(d_ckpt_num, base_err3, color = "k", label="base")
l_min = min([min(g_err3), min(d_err3)])*0.999
l_max = max([max(g_err3), max(d_err3)])*1.001
# ax2.set_yticks(np.linspace(l_min, l_max, 5))
ax2.set_xlim(0, max(d_ckpt_num))
ax2.set_ylim(l_min, l_max)
ax2.set_title('err3')
g_train_fill_upper, g_train_fill_lower = make_upper_lower(g_train_step, l_max, l_min)
ax2.fill_between(train_step, g_train_fill_lower, g_train_fill_upper, facecolor='red', alpha=alpha)
ax2.legend(loc='upper left')

ax3 = fig.add_subplot(223)
ax3.plot(g_ckpt_num, g_err5, color = "g", marker='o', label="g_longtail")
ax3.plot(d_ckpt_num, d_err5, color = "b", marker='o', label="d_longtail")
ax3.scatter(d_ckpt_num, base_err5, color = "k", label="base")
l_min = min([min(g_err5), min(d_err5)])*0.999
l_max = max([max(g_err5), max(d_err5)])*1.001
# ax3.set_yticks(np.linspace(l_min, l_max, 5))
ax3.set_ylim(l_min, l_max)
ax3.set_xlim(0, max(d_ckpt_num))
ax3.set_title('err5')
g_train_fill_upper, g_train_fill_lower = make_upper_lower(g_train_step, l_max, l_min)
ax3.fill_between(train_step, g_train_fill_lower, g_train_fill_upper, facecolor='red', alpha=alpha)
ax3.legend(loc='upper left')

ax4 = fig.add_subplot(224)
ax4.plot(g_ckpt_num, g_dcg5, color = "g", marker='o', label="g_longtail")
ax4.plot(d_ckpt_num, d_dcg5, color = "b", marker='o', label="d_longtail")
ax4.scatter(d_ckpt_num, base_dcg5, color = "k", label="base")
l_min = min([min(g_dcg5), min(d_dcg5)])*0.999
l_max = max([max(g_dcg5), max(d_dcg5)])*1.001
# ax4.set_yticks(np.linspace(l_min, l_max, 5))
ax4.set_ylim(l_min, l_max)
ax4.set_xlim(0, max(d_ckpt_num))
ax4.set_title('dcg5')
g_train_fill_upper, g_train_fill_lower = make_upper_lower(g_train_step, l_max, l_min)
ax4.fill_between(train_step, g_train_fill_lower, g_train_fill_upper, facecolor='red', alpha=alpha)
ax4.legend(loc='upper left')

plt.show()