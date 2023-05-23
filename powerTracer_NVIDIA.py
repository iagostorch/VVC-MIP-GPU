import os
import sys
from multiprocessing import Process, Event
import subprocess
import signal


def run_smi(event, traceName):
	proc = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,name,driver_version,pstate,power.draw", "--format=csv", "-lms", "1", "--filename=%s.csv" % (traceName), "&"])
	print("TRACE %s " % (traceName))
	#print("SMI PID %d" % (proc.pid))
	while not event.is_set():
		pass

	# Wait for a moment and kill nvidia-smi
	os.system("sleep 0.2")
	os.kill(proc.pid, signal.SIGTERM) 
	

def run_kernel(event, exe, refSamples, mipCosts, nFrames, reportName):
	cmd = "%s GPU 0 %s %s %s > %s.txt" % (exe, refSamples, mipCosts, nFrames, reportName)
	os.system(cmd)
	#os.system("sleep 2")
	event.set()



if __name__=='__main__':
	
	a = 1
	
	# 1080
	exe = "./main_1080"
	refSamples = "data/Cactus_BQTerrace_original_0_2x.csv"
	mipCosts_preffix = "Cactus_BQTerrace_CTU16_"
	report_preffix = "Report_1080_nFrames"
	trace_preffix = "PowerTrace_1080_nFrames"

	for n in range(1,17):
		nFrames = str(n)
		mipCosts = mipCosts_preffix + nFrames
		report = report_preffix + nFrames
		trace = trace_preffix + nFrames
		print(trace)

		event = Event() # the event is unset when created

		p1 = Process(target=run_smi, args=(event, trace, ))
		p1.start()
		p2 = Process(target=run_kernel, args=(event, exe, refSamples, mipCosts, nFrames, report, ))
		p2.start()
		
		p1.join()
		p2.join()
		


	# 4k	
	exe = "./main_4k"
	refSamples = "data/Tango2_Campfire_original_0_10x.csv"
	mipCosts_preffix = "Tango_Campfire_CTU16_"
	report_preffix = "Report_4k_nFrames"
	trace_preffix = "PowerTrace_4k_nFrames"

	for n in range(1,5):
		nFrames = str(n)
		mipCosts = mipCosts_preffix + nFrames
		report = report_preffix + nFrames
		trace = trace_preffix + nFrames
		print(trace)

		event = Event() # the event is unset when created

		p1 = Process(target=run_smi, args=(event, trace, ))
		p1.start()
		p2 = Process(target=run_kernel, args=(event, exe, refSamples, mipCosts, nFrames, report, ))
		p2.start()
		
		p1.join()
		p2.join()
		