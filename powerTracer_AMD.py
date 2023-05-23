import os
import sys
from multiprocessing import Process, Event
import subprocess
import signal


def run_smi(event, traceName):
	cmd = "rm %s.csv" % (traceName)
	os.system(cmd)
	cmd = "echo timestamp,card,power > %s.csv"  % (traceName)
	os.system(cmd)
	
	cmd = "./powerTracer_AMD_util.sh %s.csv" % (traceName)
	while not event.is_set():
		os.system(cmd)


def run_kernel(event, exe, refSamples, mipCosts, nFrames, reportName):
	cmd = "%s GPU 0 %s %s %s > %s.txt" % (exe, refSamples, mipCosts, nFrames, reportName)
	os.system(cmd)
	#os.system("sleep 2")
	event.set()



if __name__=='__main__':
	
	a = 1
	
	# 1080
	exe = "./main_1080"
	refSamples = "data/Cactus_BQTerrace_original_0_32x.csv"
	mipCosts_preffix = "Cactus_BQTerrace_CTU16_"
	report_preffix = "Report_1080_nFrames"
	trace_preffix = "PowerTrace_1080_nFrames"

	for n in range(1,21):
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

	for n in range(1,6):
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
		