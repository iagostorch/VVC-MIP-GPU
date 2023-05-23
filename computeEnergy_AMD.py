import sys
import pandas as pd
from datetime import datetime
import numpy as np
# import matplotlib.pyplot as plt


DEBUG = 0
PLOT = 1


reportCL = sys.argv[1]
reportSmi = sys.argv[2]

print("Input,%s,%s" % (reportCL, reportSmi))

start_write = ''
finish_write = ''

start_initBoundaries = ''
finish_initBoundaries = ''

start_reducedPred = ''
finish_reducedPred = ''

start_upsampleId2 = ''
finish_upsampleId2 = ''

start_upsampleId1 = ''
finish_upsampleId1 = ''

start_upsampleId0 = ''
finish_upsampleId0 = ''

start_read = ''
finish_read = ''

start_host = ''

## START TRACING THE START AND END TIMES OF EACH STAGE FROM THE CL REPORT
f = open(reportCL)
format_string = "%H:%M:%S.%f"
for line in f:
	if('STARTED HOST ' in line):
		start_host = line.split('@')[1].strip(' \n\t')
		start_host = datetime.strptime(start_host, format_string)

	if('START WRITE SAMPLES MEMOBJ' in line):
		start_write = line.split('@')[1].strip(' \n\t')
		start_write = datetime.strptime(start_write, format_string)
	if('FINISH WRITE SAMPLES MEMOBJ' in line):
		finish_write = line.split('@')[1].strip(' \n\t')
		finish_write = datetime.strptime(finish_write, format_string)

	if('START ENQUEUE initBoundaries' in line):
		start_initBoundaries = line.split('@')[1].strip(' \n\t')
		start_initBoundaries = datetime.strptime(start_initBoundaries, format_string)
	if('FINISH ENQUEUE initBoundaries' in line):
		finish_initBoundaries = line.split('@')[1].strip(' \n\t')
		finish_initBoundaries = datetime.strptime(finish_initBoundaries, format_string)

	if('START ENQUEUE reducedPred' in line):
		start_reducedPred = line.split('@')[1].strip(' \n\t')
		start_reducedPred = datetime.strptime(start_reducedPred, format_string)
	if('FINISH ENQUEUE reducedPred' in line):
		finish_reducedPred = line.split('@')[1].strip(' \n\t')
		finish_reducedPred = datetime.strptime(finish_reducedPred, format_string)

	if('START ENQUEUE upsamplePred_SIZEID=2' in line):
		start_upsampleId2 = line.split('@')[1].strip(' \n\t')
		start_upsampleId2 = datetime.strptime(start_upsampleId2, format_string)
	if('FINISH ENQUEUE upsamplePred_SIZEID=2' in line):
		finish_upsampleId2 = line.split('@')[1].strip(' \n\t')
		finish_upsampleId2 = datetime.strptime(finish_upsampleId2, format_string)

	if('START ENQUEUE upsamplePred_SIZEID=1' in line):
		start_upsampleId1 = line.split('@')[1].strip(' \n\t')
		start_upsampleId1 = datetime.strptime(start_upsampleId1, format_string)
	if('FINISH ENQUEUE upsamplePred_SIZEID=1' in line):
		finish_upsampleId1 = line.split('@')[1].strip(' \n\t')
		finish_upsampleId1 = datetime.strptime(finish_upsampleId1, format_string)

	if('START ENQUEUE upsamplePred_SIZEID=0' in line):
		start_upsampleId0 = line.split('@')[1].strip(' \n\t')
		start_upsampleId0 = datetime.strptime(start_upsampleId0, format_string)
	if('FINISH ENQUEUE upsamplePred_SIZEID=0' in line):
		finish_upsampleId0 = line.split('@')[1].strip(' \n\t')
		finish_upsampleId0 = datetime.strptime(finish_upsampleId0, format_string)

	if('START READ DISTORTION' in line):
		start_read = line.split('@')[1].strip(' \n\t')
		start_read = datetime.strptime(start_read, format_string)
	if('FINISH READ DISTORTION' in line):
		finish_read = line.split('@')[1].strip(' \n\t')
		finish_read = datetime.strptime(finish_read, format_string)

f.close()

# GET THE DELTA BETWEEN STARTING THE HOST AND THE IMPORTANT EVENTS
delta_start_host = (start_host - start_host).total_seconds()*1000
delta_start_write = (start_write - start_host).total_seconds()*1000
delta_finish_write = (finish_write - start_host).total_seconds()*1000
delta_start_initBoundaries = (start_initBoundaries - start_host).total_seconds()*1000
delta_finish_initBoundaries = (finish_initBoundaries - start_host).total_seconds()*1000
delta_start_reducedPred = (start_reducedPred - start_host).total_seconds()*1000
delta_finish_reducedPred = (finish_reducedPred - start_host).total_seconds()*1000
delta_start_upsampleId2 = (start_upsampleId2 - start_host).total_seconds()*1000
delta_finish_upsampleId2 = (finish_upsampleId2 - start_host).total_seconds()*1000
delta_start_upsampleId1 = (start_upsampleId1 - start_host).total_seconds()*1000
delta_finish_upsampleId1 = (finish_upsampleId1 - start_host).total_seconds()*1000
delta_start_upsampleId0 = (start_upsampleId0 - start_host).total_seconds()*1000
delta_finish_upsampleId0 = (finish_upsampleId0 - start_host).total_seconds()*1000
delta_start_read = (start_read - start_host).total_seconds()*1000
delta_finish_read = (finish_read - start_host).total_seconds()*1000

if(DEBUG):
	print("start_host,%d ms" % (delta_start_host))
	print("start_write,%d ms" % (delta_start_write))
	print("finish_write,%d ms" % (delta_finish_write))
	print("start_initBoundaries,%d ms" % (delta_start_initBoundaries))
	print("finish_initBoundaries,%d ms" % (delta_finish_initBoundaries))
	print("start_reducedPred,%d ms" % (delta_start_reducedPred))
	print("finish_reducedPred,%d ms" % (delta_finish_reducedPred))
	print("start_upsampleId2,%d ms" % (delta_start_upsampleId2))
	print("finish_upsampleId2,%d ms" % (delta_finish_upsampleId2))
	print("start_upsampleId1,%d ms" % (delta_start_upsampleId1))
	print("finish_upsampleId1,%d ms" % (delta_finish_upsampleId1))
	print("start_upsampleId0,%d ms" % (delta_start_upsampleId0))
	print("finish_upsampleId0,%d ms" % (delta_finish_upsampleId0))
	print("start_read,%d ms" % (delta_start_read))
	print("finish_read,%d ms" % (delta_finish_read))

## NOW READ AND PARSE THE POWER TRACE
df = pd.read_csv(reportSmi)

df = df.drop('card',axis=1)
df[['power']] = df[['power']].apply(pd.to_numeric)
df['timestamp'] = df['timestamp'].str.replace(r'\d+\-\d+\-\d+ ', '')
df['timestamp'] = pd.to_datetime(df['timestamp'])

new_df = df.copy()
start_host = start_host.replace( year=new_df['timestamp'].iloc[0].year, month=new_df['timestamp'].iloc[0].month, day=new_df['timestamp'].iloc[0].day)
# print(new_df)

# transform all timestamps into deltas in relation to host started, and convert into miliseconds (*1000)
new_df['timestamp'] = new_df['timestamp']-start_host
new_df['timestamp'] = new_df['timestamp'].dt.total_seconds()*1000

# get the average power between host start and read end
df_one_cycle = new_df[(new_df['timestamp']>=delta_start_write) & (new_df['timestamp']<=delta_finish_read)]
avg_power = df_one_cycle['power'].mean()
print('AVERAGE POWER ' + str(avg_power))
#print('AVERAGE POWER %d' % (avg_power))

if(DEBUG):
	print('Timespan between HOST_START and READ_END %dms' % (delta_finish_read))

ActiveGpuTime = (finish_read - start_write).total_seconds()*1000 #in ms
print('Timespan between WRITE_START and READ_END %dms' % (ActiveGpuTime))
energy = (avg_power*ActiveGpuTime*0.001)
print('Total energy between WRITE_START and READ_FINISH %f' % energy)

# if(PLOT):
# 	plt.plot(new_df['timestamp'], new_df[' power'], linestyle='-', linewidth=5)
# 	plt.plot()
# 	plt.xlabel('ms since host start')
# 	plt.ylabel('power [W]')
# 	#title = sys.argv[2].split('.')[0] + ' Avg Power=' + str(avg_power) + 'W ActiveGpuTime=' + str(ActiveGpuTime) + 'ms Energy=' + str(energy) + ' Joules'
# 	title = '%s AvgPower=%.2fW ActiveGpuTime=%.2fms Energy=%.2fJ' % (sys.argv[2].split('.')[0], avg_power, ActiveGpuTime, energy)
# 	plt.title(title)
# 	plt.xlim([0,1.1*delta_finish_read])
# 	# draw vertical lines for start/end of operations
# 	plt.axvline(x=delta_start_host, c='r')
# 	plt.axvline(x=delta_start_write, c='r')
# 	plt.axvline(x=delta_finish_write, c='r')
# 	plt.axvline(x=delta_start_initBoundaries, c='r')
# 	plt.axvline(x=delta_finish_initBoundaries, c='r')
# 	plt.axvline(x=delta_start_reducedPred, c='r')
# 	plt.axvline(x=delta_finish_reducedPred, c='r')
# 	plt.axvline(x=delta_start_upsampleId2, c='r')
# 	plt.axvline(x=delta_finish_upsampleId2, c='r')
# 	plt.axvline(x=delta_start_upsampleId1, c='r')
# 	plt.axvline(x=delta_finish_upsampleId1, c='r')
# 	plt.axvline(x=delta_start_upsampleId0, c='r')
# 	plt.axvline(x=delta_finish_upsampleId0, c='r')
# 	plt.axvline(x=delta_start_read, c='r')
# 	plt.axvline(x=delta_finish_read, c='r')
# 	# draw horizontal line for average power of one cycle
# 	#plt.axhline(y=avg_power, c='g', linestyle='--')
# 	plt.plot([delta_start_write, delta_finish_read],[avg_power, avg_power], 'go', linestyle='--')
# 	plt.show()