#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 

#include "constants.h"

using namespace std;

float writeTime = 0;
float readTime = 0;
float readTime_reducedBoundaries = 0;
float readTime_reducedPrediction = 0;
float execTime_reducedBoundaries = 0;
float execTime_reducedPrediction = 0;
float execTime = 0;

void probe_error(cl_int error, char* message){
    if (error != CL_SUCCESS ) {
        printf("Code %d, %s", error, message);
        return;
    }
}

// Read data from memory objects into arrays
void readMemobjsIntoArray_boundaries(cl_command_queue command_queue, int nCTUs, cl_mem redT_64x64_memObj, cl_mem redL_64x64_memObj, cl_mem redT_32x32_memObj, cl_mem redL_32x32_memObj, cl_mem redT_16x16_memObj, cl_mem redL_16x16_memObj, short *return_redT_64x64, short *return_redL_64x64, short *return_redT_32x32, short *return_redL_32x32, short *return_redT_16x16, short *return_redL_16x16){
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    error =  clEnqueueReadBuffer(command_queue, redT_64x64_memObj, CL_TRUE, 0, 
            nCTUs * 4 * cusPerCtu[_64x64] * sizeof(cl_short), return_redT_64x64, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, redL_64x64_memObj, CL_TRUE, 0, 
        nCTUs * 4 * cusPerCtu[_64x64] * sizeof(cl_short), return_redL_64x64, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    // -----------------------------

    error =  clEnqueueReadBuffer(command_queue, redT_32x32_memObj, CL_TRUE, 0, 
            nCTUs * 4 * cusPerCtu[_32x32] * sizeof(cl_short), return_redT_32x32, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, redL_32x32_memObj, CL_TRUE, 0, 
        nCTUs * 4 * cusPerCtu[_32x32] * sizeof(cl_short), return_redL_32x32, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    // -----------------------------

    error =  clEnqueueReadBuffer(command_queue, redT_16x16_memObj, CL_TRUE, 0, 
            nCTUs * 4 * cusPerCtu[_16x16] * sizeof(cl_short), return_redT_16x16, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, redL_16x16_memObj, CL_TRUE, 0, 
        nCTUs * 4 * cusPerCtu[_16x16] * sizeof(cl_short), return_redL_16x16, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;        

    readTime_reducedBoundaries = nanoSeconds;
}


// Read data from memory objects into arrays
void readMemobjsIntoArray_reducedPrediction(cl_command_queue command_queue, int nCTUs, int nPredictionModes, cl_mem reducedPredictionSignal_memObj,  short *return_reducedPredictionSignal){
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    error =  clEnqueueReadBuffer(command_queue, reducedPredictionSignal_memObj, CL_TRUE, 0, 
            nCTUs * TOTAL_CUS_PER_CTU * 8 * 8 * 12 * sizeof(cl_short), return_reducedPredictionSignal, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds = read_time_end-read_time_start;
  
    readTime_reducedPrediction = nanoSeconds;
}


// Read data from memory objects into arrays
void readMemobjsIntoArray(cl_command_queue command_queue, int numPredictions, int blockSize, cl_mem return_prediction_memObj, cl_mem return_SATD_memObj, cl_mem return_SAD_memObj, short *return_prediction, long *return_SATD, long *return_SAD, cl_mem debug_memObj, short* debug_data){
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

    error =  clEnqueueReadBuffer(command_queue, debug_memObj, CL_TRUE, 0, 
        12*64*4 * sizeof(cl_short), debug_data, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return SAD\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;
    probe_error(error, (char*)"Error reading returned memory objects into malloc'd arrays\n");
}

void reportTimingResults(){
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    printf("TIMING RESULTS (nanoseconds)\n");
    printf("Write,%f\n", writeTime);
    printf("ReducedBoundaries: Execution,%f\n", execTime_reducedBoundaries);
    printf("ReducedBoundaries: Read,%f\n", readTime_reducedBoundaries);
    printf("ReducedPrediction: Execution,%f\n", execTime_reducedPrediction);
    printf("ReducedPrediction: Read,%f\n", readTime_reducedPrediction);    
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n");    
}
