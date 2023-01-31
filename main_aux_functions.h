#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <math.h>

#include "constants.h"

using namespace std;

float writeTime = 0;
float readTime = 0;
float readTime_reducedBoundariesSize = 0;
float readTime_reducedBoundariesUnified = 0;
float readTime_reducedPrediction = 0;
float readTime_completeBoundariesSize = 0;
float readTime_completeBoundariesUnified = 0;
float readTime_SAD = 0.0;
float execTime_reducedBoundaries = 0;
float execTime_reducedPrediction = 0;
float execTime_upsampleDistortion = 0;

float execTime = 0;

void probe_error(cl_int error, char* message){
    if (error != CL_SUCCESS ) {
        printf("Code %d, %s", error, message);
        return;
    }
}

const char* translateCuSizeIdx(int cuSize){
    if(cuSize==_64x64)
        return "64x64";
    else if(cuSize==_32x32)
        return "32x32";
    
    else if(cuSize==_32x16)
        return "32x16";
    else if(cuSize==_16x32)
        return "16x32";   

    else if(cuSize==_32x8)
        return "32x8";
    else if(cuSize==_8x32)
        return "8x32";

    else if(cuSize==_16x16)
        return "16x16";

    else if(cuSize==_16x8)
        return "16x8";
    else if(cuSize==_8x16)
        return "8x16";

    else    
        return "ERROR";
}


const char* translateCuSizeIdx_NA(int cuSize){
    if(cuSize==_NA_32x16)
        return "NA_32x16";
    else if(cuSize==_NA_16x32)
        return "NA_16x32";
    else if(cuSize==_NA_32x8_G1)
        return "NA_32x8_G1";
    else if(cuSize==_NA_32x8_G2)
        return "NA_32x8_G2";
    else if(cuSize==_NA_8x32_G1)
        return "NA_8x32_G1";
    else if(cuSize==_NA_8x32_G2)
        return "NA_8x32_G2";

    else if(cuSize==_NA_16x16_G1)
        return "NA_16x16_G1";
    else if(cuSize==_NA_16x16_G2)
        return "NA_16x16_G2";
    else if(cuSize==_NA_16x16_G3)
        return "NA_16x16_G3";

    else if(cuSize==_NA_16x8_G1)
        return "NA_16x8_G1";
    else if(cuSize==_NA_16x8_G2)
        return "NA_16x8_G2";
    else if(cuSize==_NA_16x8_G3)
        return "NA_16x8_G3";
    else if(cuSize==_NA_16x8_G4)
        return "NA_16x8_G4";
    else if(cuSize==_NA_16x8_G5)
        return "NA_16x8_G5";

    else if(cuSize==_NA_8x16_G1)
        return "NA_8x16_G1";
    else if(cuSize==_NA_8x16_G2)
        return "NA_8x16_G2";
    else if(cuSize==_NA_8x16_G3)
        return "NA_8x16_G3";
    else if(cuSize==_NA_8x16_G4)
        return "NA_8x16_G4";
    else if(cuSize==_NA_8x16_G5)
        return "NA_8x16_G5";

    else    
        return "ERROR";
}

const char* translateCuSizeIdx_ALL(int cuSize){
    // ALIGNED
    if(cuSize==ALL_AL_64x64)
        return "ALL_AL_64x64";
    else if(cuSize==ALL_AL_32x32)
        return "ALL_AL_32x32";
    else if(cuSize==ALL_AL_32x16)
        return "ALL_AL_32x16";
    else if(cuSize==ALL_AL_16x32)
        return "ALL_AL_16x32";
    else if(cuSize==ALL_AL_32x8)
        return "ALL_AL_32x8";
    else if(cuSize==ALL_AL_8x32)
        return "ALL_AL_8x32";
    else if(cuSize==ALL_AL_16x16)
        return "ALL_AL_16x16";
    else if(cuSize==ALL_AL_16x8)
        return "ALL_AL_16x8";
    else if(cuSize==ALL_AL_8x16)
        return "ALL_AL_8x16";    

    // NOT ALIGNED
    else if(cuSize==ALL_NA_32x16)
        return "ALL_NA_32x16";
    else if(cuSize==ALL_NA_16x32)
        return "ALL_NA_16x32";
    else if(cuSize==ALL_NA_32x8_G1)
        return "ALL_NA_32x8_G1";
    else if(cuSize==ALL_NA_32x8_G2)
        return "ALL_NA_32x8_G2";
    else if(cuSize==ALL_NA_8x32_G1)
        return "ALL_NA_8x32_G1";
    else if(cuSize==ALL_NA_8x32_G2)
        return "ALL_NA_8x32_G2";
    else if(cuSize==ALL_NA_16x16_G1)
        return "ALL_NA_16x16_G1";
    else if(cuSize==ALL_NA_16x16_G2)
        return "ALL_NA_16x16_G2";
    else if(cuSize==ALL_NA_16x16_G3)
        return "ALL_NA_16x16_G3";
    else if(cuSize==ALL_NA_16x8_G1)
        return "ALL_NA_16x8_G1";
    else if(cuSize==ALL_NA_16x8_G2)
        return "ALL_NA_16x8_G2";
    else if(cuSize==ALL_NA_16x8_G3)
        return "ALL_NA_16x8_G3";
    else if(cuSize==ALL_NA_16x8_G4)
        return "ALL_NA_16x8_G4";
    else if(cuSize==ALL_NA_16x8_G5)
        return "ALL_NA_16x8_G5";
    else if(cuSize==ALL_NA_8x16_G1)
        return "ALL_NA_8x16_G1";
    else if(cuSize==ALL_NA_8x16_G2)
        return "ALL_NA_8x16_G2";
    else if(cuSize==ALL_NA_8x16_G3)
        return "ALL_NA_8x16_G3";
    else if(cuSize==ALL_NA_8x16_G4)
        return "ALL_NA_8x16_G4";
    else if(cuSize==ALL_NA_8x16_G5)
        return "ALL_NA_8x16_G5";

    // SizeId=1
    else if(cuSize==ALL_AL_32x4)
        return "ALL_AL_32x4";
    else if(cuSize==ALL_AL_4x32)
        return "ALL_AL_4x32";
    else if(cuSize==ALL_AL_16x4)
        return "ALL_AL_16x4";
    else if(cuSize==ALL_AL_4x16)
        return "ALL_AL_4x16";    
    else if(cuSize==ALL_AL_8x8)
        return "ALL_AL_8x8";
    else if(cuSize==ALL_AL_8x4_1half)
        return "ALL_AL_8x4_1half";
    else if(cuSize==ALL_AL_8x4_2half)
        return "ALL_AL_8x4_2half";
    else if(cuSize==ALL_AL_4x8_1half)
        return "ALL_AL_4x8_1half";
    else if(cuSize==ALL_AL_4x8_2half)
        return "ALL_AL_4x8_2half";
    else if(cuSize==ALL_NA_16x4_G123)
        return "ALL_NA_16x4_G123";
    else if(cuSize==ALL_NA_4x16_G123)
        return "ALL_NA_4x16_G123";
    else if(cuSize==ALL_NA_8x8_G1)
        return "ALL_NA_8x8_G1";
    else if(cuSize==ALL_NA_8x8_G2)
        return "ALL_NA_8x8_G2";
    else if(cuSize==ALL_NA_8x8_G3)
        return "ALL_NA_8x8_G3";
    else if(cuSize==ALL_NA_8x8_G4)
        return "ALL_NA_8x8_G4";
    else if(cuSize==ALL_NA_8x8_G5)
        return "ALL_NA_8x8_G5";
    else if(cuSize==ALL_NA_8x4_G1)
        return "ALL_NA_8x4_G1";
    else if(cuSize==ALL_NA_4x8_G1)
        return "ALL_NA_4x8_G1";

    else    
        return "ERROR";
}


// Read data from memory objects into arrays
void readMemobjsIntoArray_ReducedBoundaries(cl_command_queue command_queue, int nCTUs, cl_mem redT_64x64_memObj, cl_mem redL_64x64_memObj, cl_mem redT_32x32_memObj, cl_mem redL_32x32_memObj, cl_mem redT_16x16_memObj, cl_mem redL_16x16_memObj, short *return_redT_64x64, short *return_redL_64x64, short *return_redT_32x32, short *return_redL_32x32, short *return_redT_16x16, short *return_redL_16x16){
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

    readTime_reducedBoundariesSize = nanoSeconds;
}


void readMemobjsIntoArray_CompleteBoundaries(cl_command_queue command_queue, int nCTUs, cl_mem refT_64x64_memObj, cl_mem refL_64x64_memObj, cl_mem refT_32x32_memObj, cl_mem refL_32x32_memObj, cl_mem refT_16x16_memObj, cl_mem refL_16x16_memObj, short *return_refT_64x64, short *return_refL_64x64, short *return_refT_32x32, short *return_refL_32x32, short *return_refT_16x16, short *return_refL_16x16){
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    error =  clEnqueueReadBuffer(command_queue, refT_64x64_memObj, CL_TRUE, 0, 
            nCTUs * 128 * cuRowsPerCtu[_64x64] * sizeof(cl_short), return_refT_64x64, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, refL_64x64_memObj, CL_TRUE, 0, 
        nCTUs * 128 * cuColumnsPerCtu[_64x64] * sizeof(cl_short), return_refL_64x64, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    // -----------------------------

    error =  clEnqueueReadBuffer(command_queue, refT_32x32_memObj, CL_TRUE, 0, 
            nCTUs * 128 * cuRowsPerCtu[_32x32] * sizeof(cl_short), return_refT_32x32, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, refL_32x32_memObj, CL_TRUE, 0, 
        nCTUs * 128 * cuColumnsPerCtu[_32x32] * sizeof(cl_short), return_refL_32x32, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    // -----------------------------

    error =  clEnqueueReadBuffer(command_queue, refT_16x16_memObj, CL_TRUE, 0, 
            nCTUs * 128 * cuRowsPerCtu[_16x16] * sizeof(cl_short), return_refT_16x16, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, refL_16x16_memObj, CL_TRUE, 0, 
        nCTUs * 128 * cuColumnsPerCtu[_16x16] * sizeof(cl_short), return_refL_16x16, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    readTime_completeBoundariesSize = nanoSeconds;
}

// Read data from memory objects into arrays
void readMemobjsIntoArray_reducedPrediction(cl_command_queue command_queue, int nCTUs, int nPredictionModes, cl_mem reducedPredictionSignal_memObj,  short *return_reducedPredictionSignal, cl_mem return_SAD_memObj, long *return_SAD){
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    error =  clEnqueueReadBuffer(command_queue, reducedPredictionSignal_memObj, CL_TRUE, 0, 
            nCTUs * ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(cl_short), return_reducedPredictionSignal, 0, NULL, &read_event);
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

void readMemobjsIntoArray_Distortion(cl_command_queue command_queue, int nCTUs, int nPredictionModes, cl_mem return_SAD_memObj, long *return_SAD, cl_mem return_SATD_memObj, long *return_SATD){
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    error =  clEnqueueReadBuffer(command_queue, return_SAD_memObj, CL_TRUE, 0, 
            nCTUs * ALL_TOTAL_CUS_PER_CTU * 12 * sizeof(cl_long), return_SAD, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, return_SATD_memObj, CL_TRUE, 0, 
            nCTUs * ALL_TOTAL_CUS_PER_CTU * 12 * sizeof(cl_long), return_SATD, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    readTime_SAD = nanoSeconds;

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

void reportAllDistortionValues_ALL(long int *SAD, long int *SATD, int nCTUs){
    printf("-=-=-=-=-=-=-=-=- DISTORTION RESULTS FOR ALL CTUs -=-=-=-=-=-=-=-=-\n");
    printf("CTU,cuSize,CU,Mode,SAD,SATD\n");
    for(int ctu=0; ctu<nCTUs; ctu++){
        for(int cuSize=0; cuSize<ALL_NUM_CU_SIZES; cuSize++){
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                    // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                    printf("%d,%s,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), cu, mode);  //  Report CU size/position info
                    printf("%ld,", SAD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                    printf("%ld\n", SATD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                }
            }
        }
    }
}

void exportAllDistortionValues_File(long int *SAD, long int *SATD, int nCTUs, int frameWidth, string outputFile){
    int ctuCols = ceil(frameWidth/128.0);

    FILE *distortionFile;
    string outputFileName = outputFile + ".csv";
    distortionFile = fopen(outputFileName.c_str(),"w");
    fprintf(distortionFile,"CTU,cuSizeName,W,H,CU,X,Y,Mode,SAD,SATD\n");

    // printf("-=-=-=-=-=-=-=-=- DISTORTION RESULTS FOR ALL CTUs -=-=-=-=-=-=-=-=-\n");
    // printf("CTU,cuSizeName,W,H,CU,X,Y,Mode,SAD,SATD\n");
    int ctuX, ctuY, cuX, cuY;
    for(int ctu=0; ctu<nCTUs; ctu++){
        ctuX = 128*(ctu%ctuCols);
        ctuY = 128*(ctu/ctuCols);
        
        for(int cuSize=0; cuSize<ALL_NUM_CU_SIZES; cuSize++){
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                cuX = ctuX + ALL_X_POS[cuSize][cu];
                cuY = ctuY + ALL_Y_POS[cuSize][cu];

                for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                    // printf("%d,%s,%d,%d,%d,%d,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);  //  Report CU size/position info
                    // printf("%ld,", SAD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                    // printf("%ld\n", SATD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);

                    fprintf(distortionFile, "%d,%s,%d,%d,%d,%d,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);
                    fprintf(distortionFile, "%ld,", SAD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                    fprintf(distortionFile, "%ld\n", SATD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);

                }
            }
        }
    }
    fclose(distortionFile);
}

void reportTargetDistortionValues_ALL(long int *SAD, long int *SATD, int nCTUs, int targetCTU){
    printf("-=-=-=-=-=-=-=-=- DISTORTION RESULTS FOR CTU %d -=-=-=-=-=-=-=-=-\n", targetCTU);
    printf("CTU,cuSize,CU,Mode,SAD,SATD\n");
    for(int cuSize=0; cuSize<ALL_NUM_CU_SIZES; cuSize++){
        for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
            for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                printf("%d,%s,%d,%d,", targetCTU, translateCuSizeIdx_ALL(cuSize), cu, mode);  //  Report CU size/position info
                printf("%ld,", SAD[ targetCTU*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                printf("%ld\n", SATD[ targetCTU*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
            }
        }
    }
}

void reportTargetDistortionValues_File(long int *SAD, long int *SATD, int targetCtu, int frameWidth, string outputFile){
    int ctu = targetCtu;
    int ctuCols = ceil(frameWidth/128.0);

    FILE *distortionFile;
    string outputFileName = outputFile + ".csv";
    distortionFile = fopen(outputFileName.c_str(),"w");
    fprintf(distortionFile,"CTU,cuSizeName,W,H,CU,X,Y,Mode,SAD,SATD\n");

    int ctuX, ctuY, cuX, cuY;
    ctuX = targetCtu%ctuCols;
    ctuY = targetCtu/ctuCols;
        
    for(int cuSize=0; cuSize<ALL_NUM_CU_SIZES; cuSize++){
        for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
            cuX = ctuX + ALL_X_POS[cuSize][cu];
            cuY = ctuY + ALL_Y_POS[cuSize][cu];

            for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                // printf("%d,%s,%d,%d,%d,%d,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);  //  Report CU size/position info
                // printf("%ld,", SAD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                // printf("%ld\n", SATD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);

                fprintf(distortionFile, "%d,%s,%d,%d,%d,%d,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);
                fprintf(distortionFile, "%ld,", SAD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                fprintf(distortionFile, "%ld\n", SATD[ ctu*ALL_TOTAL_CUS_PER_CTU*12 + ALL_stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);

            }
        }
    }
    fclose(distortionFile);
}

void reportTimingResults(){
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    printf("TIMING RESULTS (nanoseconds)\n");
    printf("Write,%f\n", writeTime);
    printf("InitBoundaries:  Execution,%f\n", execTime_reducedBoundaries);
    printf("InitBoundaries:  Read size-reduced,%f\n", readTime_reducedBoundariesSize);
    printf("InitBoundaries:  Read size-complete,%f\n", readTime_completeBoundariesSize);
    printf("InitBoundaries:  Read unified-reduced,%f\n", readTime_reducedBoundariesUnified);
    printf("InitBoundaries:  Read unified-complete,%f\n", readTime_completeBoundariesUnified);

    printf("ReducedPrediction:  Execution,%f\n", execTime_reducedPrediction);
    printf("ReducedPrediction:  Read,%f\n", readTime_reducedPrediction);   

    printf("UpsamplePredictionDistortion: Execution,%f\n", execTime_upsampleDistortion); 
    printf("UpsamplePredictionDistortion: Read,%f\n", readTime_SAD); 
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n");    
}


void readMemobjsIntoArray_UnifiedBoundaries(cl_command_queue command_queue, int nCTUs, cl_mem redT_all_memObj, cl_mem refT_all_memObj, short *return_unified_redT, short *return_unified_refT, cl_mem redL_all_memObj, cl_mem refL_all_memObj, short *return_unified_redL, short *return_unified_refL){
    
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    error =  clEnqueueReadBuffer(command_queue, redT_all_memObj, CL_TRUE, 0, 
            nCTUs * ALL_TOTAL_CUS_PER_CTU * 4 * sizeof(cl_short), return_unified_redT, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return reduced boundaries\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, redL_all_memObj, CL_TRUE, 0, 
            nCTUs * ALL_TOTAL_CUS_PER_CTU * 4 * sizeof(cl_short), return_unified_redL, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return reduced boundaries\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    readTime_reducedBoundariesUnified = nanoSeconds;
    nanoSeconds = 0.0;


    error =  clEnqueueReadBuffer(command_queue, refT_all_memObj, CL_TRUE, 0, 
            nCTUs * ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES] * sizeof(cl_short), return_unified_refT, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return complete boundaries\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, refL_all_memObj, CL_TRUE, 0, 
            nCTUs * ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES] * sizeof(cl_short), return_unified_refL, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return complete boundaries\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    readTime_completeBoundariesUnified = nanoSeconds;
    nanoSeconds = 0.0;
}


void reportReducedBoundariesTargetCtu_ALL(short *unified_redT, short *unified_redL, int targetCTU, int frameWidth, int frameHeight){
    printf("=-=-=-=-=- UNIFIED RESULTS FOR CTU %d @(%dx%d)\n", targetCTU, 128 * (targetCTU % (int) ceil(frameWidth/128)), 128 * (targetCTU / (int) ceil(frameWidth/128)));
    
    printf("=-=-=-=-=- REDUCED TOP BOUNDARIES RESULTS -=-=-=-=-=\n");
    for(int cuSizeIdx=0; cuSizeIdx<ALL_NUM_CU_SIZES; cuSizeIdx++){
        int boundarySize = ALL_reducedBoundarySizes[cuSizeIdx];
        // TODO: For SizeId=0 the boundaries are small and we will need a different way to address this
        if(boundarySize>2){
            printf("RESULTS FOR %s\n", translateCuSizeIdx_ALL(cuSizeIdx));
            for (int cu = 0; cu < ALL_cusPerCtu[cuSizeIdx]; cu++){
                printf("CU %d\n", cu);
                for(int b=0; b<boundarySize; b++){
                    printf("%d,", unified_redT[targetCTU*ALL_TOTAL_CUS_PER_CTU*boundarySize + ALL_stridedCusPerCtu[cuSizeIdx]*boundarySize + cu*boundarySize + b]);
                }
                printf("\n");
            }
            printf("\n");
        }

    }

    printf("=-=-=-=-=- REDUCED LEFT BOUNDARIES RESULTS -=-=-=-=-=\n");
    for(int cuSizeIdx=0; cuSizeIdx<ALL_NUM_CU_SIZES; cuSizeIdx++){
        int boundarySize = ALL_reducedBoundarySizes[cuSizeIdx];
        // TODO: For SizeId=0 the boundaries are small and we will need a different way to address this
        if(boundarySize>2){
            printf("RESULTS FOR %s\n", translateCuSizeIdx_ALL(cuSizeIdx));
            for (int cu = 0; cu < ALL_cusPerCtu[cuSizeIdx]; cu++){
                printf("CU %d\n", cu);
                for(int b=0; b<boundarySize; b++){
                    printf("%d,", unified_redL[targetCTU*ALL_TOTAL_CUS_PER_CTU*boundarySize + ALL_stridedCusPerCtu[cuSizeIdx]*boundarySize + cu*boundarySize + b]);
                }
                printf("\n");
            }
            printf("\n");
        }
    }
}

void reportCompleteBoundariesTargetCtu_ALL(short *unified_refT, short *unified_refL, int targetCTU, int frameWidth, int frameHeight){
    printf("=-=-=-=-=- UNIFIED RESULTS FOR CTU %d @(%dx%d)\n", targetCTU, 128 * (targetCTU % (int) ceil(frameWidth/128)), 128 * (targetCTU / (int) ceil(frameWidth/128)));
    
    printf("=-=-=-=-=- COMPLETE TOP BOUNDARIES RESULTS -=-=-=-=-=\n");
    for(int cuSizeIdx=0; cuSizeIdx<ALL_NUM_CU_SIZES; cuSizeIdx++){
        printf("RESULTS FOR %s\n", translateCuSizeIdx_ALL(cuSizeIdx));
        for (int cu = 0; cu < ALL_cusPerCtu[cuSizeIdx]; cu++){
            printf("CU %d\n", cu);
            for(int sample=0; sample<ALL_widths[cuSizeIdx]; sample++){
                printf("%d,", unified_refT[targetCTU*ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES] + ALL_stridedCompleteTopBoundaries[cuSizeIdx] + cu*ALL_widths[cuSizeIdx] + sample]);
                            //   unified_refT[ctuIdx*   ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES] + ALL_stridedCompleteTopBoundaries[cuSizeIdx] + (row*cuColumnsPerCtu+col)*cuWidth + w]);
            }
            printf("\n");
        }
        printf("\n");
    }

    for(int cuSizeIdx=0; cuSizeIdx<ALL_NUM_CU_SIZES; cuSizeIdx++){
        printf("RESULTS FOR %s\n", translateCuSizeIdx_ALL(cuSizeIdx));
        for (int cu = 0; cu < ALL_cusPerCtu[cuSizeIdx]; cu++){
            printf("CU %d\n", cu);
            for(int sample=0; sample<ALL_heights[cuSizeIdx]; sample++){
                printf("%d,", unified_refL[targetCTU*ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES] + ALL_stridedCompleteLeftBoundaries[cuSizeIdx] + cu*ALL_heights[cuSizeIdx] + sample]);
            }
            printf("\n");
        }
        printf("\n");
    }
}


void reportReducedPredictionTargetCtu_ALL(short *reducedPrediction, int targetCTU, int frameWidth, int frameHeight){
    printf("TRACING REDUCED PREDICTION SIGNAL FOR CTU %d\n", targetCTU);
    for(int cuSizeIdx=0; cuSizeIdx<ALL_NUM_CU_SIZES; cuSizeIdx++){
        int predSize = ALL_reducedPredSizes[cuSizeIdx];
        int numModes = ALL_numPredModes[cuSizeIdx];
        printf("      RESULTS FOR CUs %s\n", translateCuSizeIdx_ALL(cuSizeIdx));
        for(int cu=0; cu<ALL_cusPerCtu[cuSizeIdx]; cu++){
            for(int m=0; m<numModes*2; m++){
                int ctuIdx = targetCTU*ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES];
                // Point to start of this CU size in global buffer
                int currCuModeIdx = ctuIdx + ALL_stridedPredictionsPerCtu[cuSizeIdx];
                // Point to start of this CU specifically in global buffer
                currCuModeIdx += cu*predSize*predSize*numModes*2;
                // Point to start of the current mode in global buffer
                currCuModeIdx += m*predSize*predSize;

                printf("===>>> Size %s  ||  CU %d, MODE %d\n", translateCuSizeIdx_ALL(cuSizeIdx), cu, m);
                for(int i=0; i<predSize; i++){
                    for(int j=0; j<predSize; j++){
                        printf("%d,", reducedPrediction[currCuModeIdx + i*predSize + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }
    }
}

void reportReducedPredictionTargetCtu(short *reducedPrediction, int targetCTU, int frameWidth, int frameHeight){
    printf("TRACING REDUCED PREDICTION SIGNAL FOR CTU %d\n", targetCTU);
    for(int cuSizeIdx=0; cuSizeIdx<NUM_CU_SIZES; cuSizeIdx++){
        printf("      RESULTS FOR CUs %s\n", translateCuSizeIdx(cuSizeIdx));
        for(int cu=0; cu<cusPerCtu[cuSizeIdx]; cu++){
            for(int m=0; m<PREDICTION_MODES_ID2*2; m++){
                int ctuIdx = targetCTU*TOTAL_CUS_PER_CTU*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
                // Point to start of this CU size in global buffer
                int currCuModeIdx = ctuIdx + stridedCusPerCtu[cuSizeIdx]*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
                // Point to start of this CU specifically in global buffer
                currCuModeIdx += cu*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
                // Point to start of the current mode in global buffer
                currCuModeIdx += m*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2;

                printf("===>>> Size %d  ||  CU %d, MODE %d\n", cuSizeIdx, cu, m);
                for(int i=0; i<8; i++){
                    for(int j=0; j<8; j++){
                        printf("%d,", reducedPrediction[currCuModeIdx + i*8 + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }
    }
}