#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
using namespace std;

float writeTime = 0;
float readTime = 0;
float execTime = 0;

void probe_error(cl_int error, char* message){
    if (error != CL_SUCCESS ) {
        printf("Code %d, %s", error, message);
        return;
    }
}

// Read data from memory objects into arrays
void readMemobjsIntoArray(cl_command_queue command_queue, int numPredictions, int blockSize, cl_mem return_prediction_memObj, cl_mem return_SATD_memObj, cl_mem return_SAD_memObj, short *return_prediction, long *return_SATD, long *return_SAD){
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    
    error =  clEnqueueReadBuffer(command_queue, return_prediction_memObj, CL_TRUE, 0, 
            numPredictions * blockSize * sizeof(cl_short), return_prediction, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, return_SATD_memObj, CL_TRUE, 0, 
            numPredictions * sizeof(cl_long), return_SATD, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return SATD\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;
    probe_error(error, (char*)"Error reading returned memory objects into malloc'd arrays\n");

    error =  clEnqueueReadBuffer(command_queue, return_SAD_memObj, CL_TRUE, 0, 
            numPredictions * sizeof(cl_long), return_SAD, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return SAD\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;
    probe_error(error, (char*)"Error reading returned memory objects into malloc'd arrays\n");

    readTime = nanoSeconds;
}

void reportTimingResults(){
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    printf("TIMING RESULTS (nanoseconds)\n");
    printf("Write,%f\n", writeTime);
    printf("Execution,%f\n", execTime);
    printf("Read,%f\n", readTime);
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n");    
}
