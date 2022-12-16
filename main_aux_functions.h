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
    else if(cuSize==_16x16)
        return "16x16";
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

void readMemobjsIntoArray_Distortion(cl_command_queue command_queue, int nCTUs, int nPredictionModes, cl_mem return_SAD_memObj, long *return_SAD, cl_mem return_SATD_memObj, long *return_SATD){
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    error =  clEnqueueReadBuffer(command_queue, return_SAD_memObj, CL_TRUE, 0, 
            nCTUs * TOTAL_CUS_PER_CTU * 12 * sizeof(cl_long), return_SAD, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, return_SATD_memObj, CL_TRUE, 0, 
            nCTUs * TOTAL_CUS_PER_CTU * 12 * sizeof(cl_long), return_SATD, 0, NULL, &read_event);
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

void reportAllDistortionValues(long int *SAD, long int *SATD, int nCTUs){
    printf("-=-=-=-=-=-=-=-=- DISTORTION RESULTS FOR ALL CTUs -=-=-=-=-=-=-=-=-\n");
    printf("CTU,cuSize,CU,Mode,SAD,SATD\n");
    for(int ctu=0; ctu<nCTUs; ctu++){
        for(int cuSize=0; cuSize<NUM_CU_SIZES; cuSize++){
            for(int cu=0; cu<cusPerCtu[cuSize]; cu++){
                for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                    // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                    printf("%d,%s,%d,%d,", ctu, translateCuSizeIdx(cuSize), cu, mode);  //  Report CU size/position info
                    printf("%ld,", SAD[ ctu*TOTAL_CUS_PER_CTU*12 + stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                    printf("%ld\n", SATD[ ctu*TOTAL_CUS_PER_CTU*12 + stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                }
            }
        }
    }
}

void reportTargetDistortionValues(long int *SAD, long int *SATD, int nCTUs, int targetCTU){
    printf("-=-=-=-=-=-=-=-=- DISTORTION RESULTS FOR CTU %d -=-=-=-=-=-=-=-=-\n", targetCTU);
    printf("CTU,cuSize,CU,Mode,SAD,SATD\n");
    for(int cuSize=0; cuSize<NUM_CU_SIZES; cuSize++){
        for(int cu=0; cu<cusPerCtu[cuSize]; cu++){
            for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                printf("%d,%s,%d,%d,", targetCTU, translateCuSizeIdx(cuSize), cu, mode);  //  Report CU size/position info
                printf("%ld,", SAD[ targetCTU*TOTAL_CUS_PER_CTU*12 + stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
                printf("%ld\n", SATD[ targetCTU*TOTAL_CUS_PER_CTU*12 + stridedCusPerCtu[cuSize]*12 + cu*12 + mode ]);
            }
        }
    }
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
            nCTUs * TOTAL_CUS_PER_CTU * 4 * sizeof(cl_short), return_unified_redT, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, redL_all_memObj, CL_TRUE, 0, 
            nCTUs * TOTAL_CUS_PER_CTU * 4 * sizeof(cl_short), return_unified_redL, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
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
            nCTUs * stridedCompleteTopBoundaries[NUM_CU_SIZES] * sizeof(cl_short), return_unified_refT, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, refL_all_memObj, CL_TRUE, 0, 
            nCTUs * stridedCompleteTopBoundaries[NUM_CU_SIZES] * sizeof(cl_short), return_unified_refL, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prefiction\n");
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

