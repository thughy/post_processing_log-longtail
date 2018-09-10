import os 
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import cmp_to_key

base = [0.512593684728,
		0.526861539905,
		0.526407936971,
		0.556288298804,
		0.501453905066]

if_read_one = False
if_g_train_first = 1.0
g_step = 2
d_step = 10000
g_gap = d_step*0.8
d_gap = g_step*0.8
average_num = 30
alpha = 0.25

if_use_old_label_flag = False
batches_in_oneckpt = int(12800000/400)
cut_off_step = 10000000
# parameters
current_dir_path = os.getcwd()
case = current_dir_path[3:].split('\\')[-1]
print('this case is ', case)
File_path = 'D:/work/models/transformer/results/'


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

# ----------------------------- start to read results log--------------------------------------
res_file2open = []
for file in os.listdir(current_dir_path):
	if 'results' in file:
		res_file2open.append(file)

g_info = []
d_info = []
steps = []

if if_read_one:
	res_file2open = res_file2open[0]
	res_file2open = [res_file2open]

class g_Info:
  def __ini__(self):
    self.step=0
    self.reward=0
    self.loss=0
    self.total_loss=0
    self.acc=0

class d_Info:
  def __ini__(self):
    self.step=0
    self.reward=0
    self.loss=0
    self.total_loss=0
    self.acc=0

for file in res_file2open:
	with open(file, 'r') as f:
		print('reading '+file)
		old_label = ''
		first_num = True
		period_start = 0
		cnt = 0

		for line in f.readlines():
			if 'worker' in line:
				char1 = 'total_step is '
				position1 = line.find(char1)
				char2 = ', d_global_step is '
				position2 = line.find(char2)
				step = float(line[position1+len(char1):position2])/batches_in_oneckpt
				steps.append(step)

				char1 = '_: '
				position1 = line.find(char1)
				char2 = ', total_step is '
				position2 = line.find(char2)
				one_label = line[position1+len(char1):position2]
				assert(one_label=='generator' or one_label=='discriminator')

				if one_label == old_label or (if_use_old_label_flag != True):
					first_num = False 
					if one_label=='generator':
						info = g_Info()
						g_info.append(info)
						g_info[-1].step = step
					else:
						info = d_Info()
						d_info.append(info)
						d_info[-1].step = step
				else:
					first_num = True

				cnt += 1

			if 'reward' in line:
				char1 = 'reward is '
				position1 = line.find(char1)
				char2 = ', loss is '
				position2 = line.find(char2)
				one_reward = float(line[position1+len(char1):position2])

				char1 = ', loss is '
				position1 = line.find(char1)
				char2 = ', regu_loss is '
				position2 = line.find(char2)
				one_loss = float(line[position1+len(char1):position2])

				char1 = ', total_loss is '
				position1 = line.find(char1)
				one_total_loss = float(line[position1+len(char1):len(line)])

				if (first_num != True) or (if_use_old_label_flag != True):
					if one_label == 'generator':
						g_info[-1].reward = one_reward
						g_info[-1].loss = one_loss
						g_info[-1].total_loss = one_total_loss
					else:
						d_info[-1].reward = one_reward
						d_info[-1].loss = one_loss
						d_info[-1].total_loss = one_total_loss

			if 'acc' in line:
				char1 = ', acc is '
				position1 = line.find(char1)
				one_acc = float(line[position1+len(char1):len(line)])
				if (first_num != True) or (if_use_old_label_flag != True):
					if one_label == 'generator':
						g_info[-1].acc = one_acc
					else:
						d_info[-1].acc = one_acc
			old_label = one_label

def myCmp(a, b):
  if a.step > b.step:
    return 1
  else:
    return -1

# ----------------------------- sort and average results log--------------------------------------
g_info.sort(key=cmp_to_key(myCmp))
d_info.sort(key=cmp_to_key(myCmp))

g_acc = []
g_steps = []
g_reward = []
g_loss = []
g_total_loss = []

d_acc = []
d_steps = []
d_reward = []
d_loss = []
d_total_loss = []

g_old_step = g_info[0].step
g_period_start = []
g_period_end = []
g_period_start.append(0)

for i in range(len(g_info)):
	g_steps.append(g_info[i].step)
	g_acc.append(g_info[i].acc)
	g_reward.append(g_info[i].reward)
	g_loss.append(g_info[i].loss)
	g_total_loss.append(g_info[i].total_loss)

	g_step_diff = g_steps[-1] - g_old_step
	if g_step_diff > g_gap:
		g_period_end.append(i-1)
		g_period_start.append(i)
	g_old_step  = g_steps[-1]

if len(g_period_start) != len(g_period_end):
	g_period_end.append(len(g_info)-1)

d_old_step = d_info[0].step
d_period_start = []
d_period_end = []
d_period_start.append(0)
for i in range(len(d_info)):
	d_steps.append(d_info[i].step)
	d_acc.append(d_info[i].acc)
	d_reward.append(d_info[i].reward)
	d_loss.append(d_info[i].loss)
	d_total_loss.append(d_info[i].total_loss)

	d_step_diff = d_steps[-1] - d_old_step
	if d_step_diff > d_gap:
		d_period_end.append(i-1)
		d_period_start.append(i)
	d_old_step  = d_steps[-1]

if len(d_period_start) != len(d_period_end):
	d_period_end.append(len(d_info)-1)

assert(len(g_period_start) == len(g_period_end))
assert(len(d_period_start) == len(d_period_end))

def average_arr1(arr):
	arr1 = []
	for i in range(average_num, len(arr), average_num):
		ave_temp = sum(arr[i-average_num:i])*1.0/average_num
		temp2 = [ave_temp for i in range(average_num)]
		arr1 += temp2
	ave_temp = sum(arr[len(arr)-average_num:len(arr)])*1.0/average_num
	last = ((len(arr)-1)//average_num)*average_num
	temp2 = [ave_temp for i in range(last, len(arr), 1)]
	arr1 += temp2
	if len(arr) != len(arr1):
		print(len(arr), len(arr1))
	assert(len(arr) == len(arr1))
	return arr1

def average_arr(arr):
	arr1 = []
	for i in range(len(arr)):
		start = max([i-int(average_num/2),0])
		end = min([i+int(average_num/2),len(arr)])
		ave_temp = sum(arr[start:end])*1.0/(end-start)
		arr1.append(ave_temp)
	return arr1

print(g_period_start)
print(g_period_end)
print(d_period_start)
print(d_period_end)

g_acc_a   = []
g_reward_a = []
g_loss_a = []
g_total_loss_a = []

d_acc_a = []
d_reward_a = []
d_loss_a = []
d_total_loss_a = []

for i in range(len(g_period_start)):
	start = g_period_start[i]
	end   = g_period_end[i]
	acc_to_aver     = g_acc[start:end+1]
	reward_to_aver  = g_reward[start:end+1]
	loss_to_aver    = g_loss[start:end+1]
	totloss_to_aver = g_total_loss[start:end+1]

	temp1 = average_arr(acc_to_aver)
	temp2 = average_arr(reward_to_aver)
	temp3 = average_arr(loss_to_aver)
	temp4 = average_arr(totloss_to_aver)

	g_acc_a += temp1
	g_reward_a += temp2
	g_loss_a += temp3
	g_total_loss_a += temp4

for i in range(len(d_period_start)):
	start = d_period_start[i]
	end   = d_period_end[i]
	acc_to_aver     = d_acc[start:end+1]
	reward_to_aver  = d_reward[start:end+1]
	loss_to_aver    = d_loss[start:end+1]
	totloss_to_aver = d_total_loss[start:end+1]

	temp1 = average_arr(acc_to_aver)
	temp2 = average_arr(reward_to_aver)
	temp3 = average_arr(loss_to_aver)
	temp4 = average_arr(totloss_to_aver)

	d_acc_a += temp1
	d_reward_a += temp2
	d_loss_a += temp3
	d_total_loss_a += temp4

# ----------------------------- read compare long tail results --------------------------------------
keyword_to_read = 'f_rank'

d_ckpt_num = []
d_err1 = []
d_err3 = []
d_err5 = []
d_dcg5 = []
d_goodpos = []
d_aver = []

g_ckpt_num = []
g_err1 = []
g_err3 = []
g_err5 = []
g_dcg5 = []
g_goodpos = []
g_aver = []

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
av = (base[0]+base[1]+base[2]+base[3])/4.0
g_aver.append(av)
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
			av = (float(res[0])+float(res[1])+float(res[2])+float(res[3]))/4.0
			g_aver.append(av)

d_ckpt_num.append(int(0))
d_err1.append(float(base[0]))
d_err3.append(float(base[1]))
d_err5.append(float(base[2]))
d_dcg5.append(float(base[3]))
av = (base[0]+base[1]+base[2]+base[3])/4.0
d_aver.append(av)
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
			av = (float(res[0])+float(res[1])+float(res[2])+float(res[3]))/4.0
			d_aver.append(av)

# start to write ckpt
# write to txt file
txtName = keyword_to_read
txtName += "_res.txt"

temp = ''
base_err1 = []
base_err3 = []
base_err5 = []
base_dcg5 = []
base_ave = []
assert(len(d_ckpt_num) == len(g_ckpt_num))

with open(txtName, "w")  as f:
	for i in range(max([len(d_ckpt_num), len(g_ckpt_num)])):
		base_err1.append(float(base[0]))
		base_err3.append(float(base[1]))
		base_err5.append(float(base[2]))
		base_dcg5.append(float(base[3]))
		av = (base[0]+base[1]+base[2]+base[3])/4.0
		base_ave.append(av)
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




# ----------------------------- plot results --------------------------------------
# 
def make_upper_lower(g_train_step, l_max, l_min):
	g_train_fill_upper = [a*l_max for a in g_train_step]
	g_train_fill_lower = [l_min for i in range(len(g_train_fill_upper))]
	for i,a in enumerate(g_train_fill_upper):
		if a < l_min:
			g_train_fill_upper[i] = l_min
	return g_train_fill_upper, g_train_fill_lower

linestyle = None
fig = plt.figure(1) 
ax1 = fig.add_subplot(311)
ax1.scatter(g_steps, g_acc, color = "g", label="g_acc")
ax1.plot(g_steps, g_acc_a, color = "k", linestyle=linestyle, label="g_acc_average")
ax1.legend(loc='upper right')
l_min = min(g_acc)
l_max = max(g_acc)
ax1.set_xlim(0, max(steps))
ax1.set_ylim(l_min, l_max)

ax2 = fig.add_subplot(312)
ax2.scatter(d_steps, d_acc, color = "b", label="d_acc")
ax2.plot(d_steps, d_acc_a, color = "k", linestyle=linestyle, label="d_acc_average")
ax2.legend(loc='upper right')
l_min = min(d_acc)
l_max = max(d_acc)
ax2.set_xlim(0, max(steps))
ax2.set_ylim(l_min, l_max)

ax3 = fig.add_subplot(313)
ax3.plot(g_ckpt_num, g_aver, color = "g", marker='o', label="g_longtail")
ax3.plot(d_ckpt_num, d_aver, color = "b", marker='o', label="d_longtail")
ax3.scatter(d_ckpt_num, base_ave, color = "k", label="base")
l_min = min([min(g_aver), min(d_aver)])*0.999
l_max = max([max(g_aver), max(d_aver)])*1.001
# ax1.set_yticks(np.linspace(l_min, l_max, 5))
ax3.set_ylim(l_min, l_max)
ax3.set_xlim(0, max(steps))
ax3.set_title('longtail')
g_train_fill_upper, g_train_fill_lower = make_upper_lower(g_train_step, l_max, l_min)
ax3.fill_between(train_step, g_train_fill_lower, g_train_fill_upper, facecolor='red', alpha=alpha)
ax3.legend(loc='upper right')

fig = plt.figure(2) 
ax1 = fig.add_subplot(311)
ax1.scatter(g_steps, g_reward, color = "g", label="g_reward")
ax1.plot(g_steps, g_reward_a, color = "k", linestyle=linestyle, label="g_reward_average")
ax1.legend(loc='upper right')
l_min = min(g_reward)
l_max = max(g_reward)
ax1.set_xlim(0, max(steps))
ax1.set_ylim(l_min, l_max)

ax2 = fig.add_subplot(312)
ax2.scatter(d_steps, d_reward, color = "b", label="d_reward")
ax2.plot(d_steps, d_reward_a, color = "k", linestyle=linestyle, label="d_reward_average")
ax2.legend(loc='upper right')
l_min = min(d_reward)
l_max = max(d_reward)
ax2.set_xlim(0, max(steps))
ax2.set_ylim(l_min, l_max)

ax3 = fig.add_subplot(313)
ax3.plot(g_ckpt_num, g_aver, color = "g", marker='o', label="g_longtail")
ax3.plot(d_ckpt_num, d_aver, color = "b", marker='o', label="d_longtail")
ax3.scatter(d_ckpt_num, base_ave, color = "k", label="base")
l_min = min([min(g_aver), min(d_aver)])*0.999
l_max = max([max(g_aver), max(d_aver)])*1.001
# ax1.set_yticks(np.linspace(l_min, l_max, 5))
ax3.set_ylim(l_min, l_max)
ax3.set_xlim(0, max(steps))
ax3.set_title('longtail')
g_train_fill_upper, g_train_fill_lower = make_upper_lower(g_train_step, l_max, l_min)
ax3.fill_between(train_step, g_train_fill_lower, g_train_fill_upper, facecolor='red', alpha=alpha)
ax3.legend(loc='upper right')


fig = plt.figure(3) 
ax1 = fig.add_subplot(311)
ax1.scatter(g_steps, g_loss, color = "g", label="g_loss")
ax1.plot(g_steps, g_loss_a, color = "k", linestyle=linestyle, label="g_loss_average")
ax1.legend(loc='upper right')
l_min = min(g_loss)
l_max = max(g_loss)
ax1.set_xlim(0, max(steps))
ax1.set_ylim(l_min, l_max)

ax2 = fig.add_subplot(312)
ax2.scatter(d_steps, d_loss, color = "b", label="d_loss")
ax2.plot(d_steps, d_loss_a, color = "k", linestyle=linestyle, label="d_loss_average")
ax2.legend(loc='upper right')
l_min = min(d_loss)
l_max = max(d_loss)
ax2.set_xlim(0, max(steps))
ax2.set_ylim(l_min, l_max)

ax3 = fig.add_subplot(313)
ax3.plot(g_ckpt_num, g_aver, color = "g", marker='o', label="g_longtail")
ax3.plot(d_ckpt_num, d_aver, color = "b", marker='o', label="d_longtail")
ax3.scatter(d_ckpt_num, base_ave, color = "k", label="base")
l_min = min([min(g_aver), min(d_aver)])*0.999
l_max = max([max(g_aver), max(d_aver)])*1.001
# ax1.set_yticks(np.linspace(l_min, l_max, 5))
ax3.set_ylim(l_min, l_max)
ax3.set_xlim(0, max(steps))
ax3.set_title('longtail')
g_train_fill_upper, g_train_fill_lower = make_upper_lower(g_train_step, l_max, l_min)
ax3.fill_between(train_step, g_train_fill_lower, g_train_fill_upper, facecolor='red', alpha=alpha)
ax3.legend(loc='upper right')


fig = plt.figure(4) 
ax1 = fig.add_subplot(311)
ax1.scatter(g_steps, g_total_loss, color = "g", label="g_total_loss")
ax1.plot(g_steps, g_total_loss_a, color = "k", linestyle=linestyle, label="g_total_loss_average")
ax1.legend(loc='upper right')
l_min = min(g_total_loss)
l_max = max(g_total_loss)
ax1.set_xlim(0, max(steps))
ax1.set_ylim(l_min, l_max)

ax2 = fig.add_subplot(312)
ax2.scatter(d_steps, d_total_loss, color = "b", label="d_total_loss")
ax2.plot(d_steps, d_total_loss_a, color = "k", linestyle=linestyle, label="d_total_loss_average")
ax2.legend(loc='upper right')
l_min = min(d_total_loss)
l_max = max(d_total_loss)
ax2.set_xlim(0, max(steps))
ax2.set_ylim(l_min, l_max)

ax3 = fig.add_subplot(313)
ax3.plot(g_ckpt_num, g_aver, color = "g", marker='o', label="g_longtail")
ax3.plot(d_ckpt_num, d_aver, color = "b", marker='o', label="d_longtail")
ax3.scatter(d_ckpt_num, base_ave, color = "k", label="base")
l_min = min([min(g_aver), min(d_aver)])*0.999
l_max = max([max(g_aver), max(d_aver)])*1.001
# ax1.set_yticks(np.linspace(l_min, l_max, 5))
ax3.set_ylim(l_min, l_max)
ax3.set_xlim(0, max(steps))
ax3.set_title('longtail')
g_train_fill_upper, g_train_fill_lower = make_upper_lower(g_train_step, l_max, l_min)
ax3.fill_between(train_step, g_train_fill_lower, g_train_fill_upper, facecolor='red', alpha=alpha)
ax3.legend(loc='upper right')

plt.show()