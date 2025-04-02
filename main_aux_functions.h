#define MAX_PERFORMANCE_DIST 1 // When enabled, the individual values of SAD and SATD are not offloaded to the host, and the host does not read into malloc'd arrays

#define TRACE_POWER 1   // When enabled the host is simplified by reducing unneccessary memory reads and prints, the timestamp of major operations is printed, the GPU operations are repeated N_FRAMES times

#define BUFFER_SLOTS 2 // Numbe of frames stored at once in memory (reference samples, distortion, and metadata)

#define MULTIPLIER_CPU_FILTER 1

int N_FRAMES = -1; // Overwritten by sys.argv

#include <CL/cl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <math.h>
#include <time.h>
#include <sys/time.h>

#include "constants.h"

#include<boost/program_options.hpp>

namespace po = boost::program_options;

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
float execTime_upsampleDistortion_SizeId2 = 0;
float execTime_upsampleDistortion_SizeId1 = 0;
float execTime_upsampleDistortion_SizeId0 = 0;
float totalGpuTime = 0; // Write, Exec, Read

float writeTime_filter = 0;
float executionTime_filter = 0;
float readTime_filter = 0;

float execTime = 0;

// used for the timestamps
typedef struct DateAndTime {
    int year;
    int month;
    int day;
    int hour;
    int minutes;
    int seconds;
    int msec;
} DateAndTime;

int computeTimeDifferenceMs(DateAndTime start, DateAndTime finish){
    int difference = 0;

    // Consider ms
    if(finish.msec>=start.msec){
        difference += finish.msec-start.msec;
    }
    else{
        finish.seconds--;
        difference += 1000 + finish.msec-start.msec;
    }

    // Consider Seconds
    if(finish.seconds>=start.seconds){
        difference += 1000 * (finish.seconds-start.seconds);
    }
    else{
        finish.minutes--;
        difference += 1000 * (60 + finish.seconds-start.seconds);
    }

    // Consider Minutes
    if(finish.minutes>=start.minutes){
        difference += 1000*60 * (finish.minutes-start.minutes);
    }
    else{
        finish.hour--;
        difference += 1000*60 * (60 + finish.minutes-start.minutes);
    }
    
    // Consider Hours
    if(finish.hour>=start.hour){
        difference += 1000*60*60 * (finish.hour-start.hour);
    }
    else{
        finish.day--;
        difference += 1000*60*60 * (24 + finish.hour-start.hour);
    }

    return difference;

}

DateAndTime date_and_time;
DateAndTime startWriteSamples, endReadingDistortion; // Used to track the processing time
struct timeval tv;
struct tm *timeStruct;

int timeDifference; // in miliseconds

int checkReportParameters(po::variables_map vm, int ALTERNATIVE_SAMPLES){
    int errors = 0;
    
    cout << "-=-= INPUT PARAMETERS =-=-" << endl;
    
    if( vm["DeviceIndex"].defaulted()){
        cout << "  Device index not set. Using standard value of " << vm["DeviceIndex"].as<int>() << "." << endl;
    }
    else{
        cout << "  Device Index=" << vm["DeviceIndex"].as<int>() << endl;
    }
    if(vm["OutputPreffix"].defaulted() ){
        cout << "  OutputPreffix log file not set. The output will not be written to any file." << endl;
    }
    else{
        cout << "  OutputPreffix=" << vm["OutputPreffix"].as<string>() <<  endl;
    }
    if( ! vm["FramesToBeEncoded"].empty() ){
        cout << "  FramesToBeEncoded=" << vm["FramesToBeEncoded"].as<int>() << endl;
    }
    else{
        cout << "  [!] ERROR: FramesToBeEncoded not set." << endl;
        errors++;
    }
    if ( ! vm["OriginalFrames"].empty() ){
        cout << "  InputOriginalFrame=" << vm["OriginalFrames"].as<string>() << endl;
    }
    else{
        cout << "  [!] ERROR: Input original frames not set." << endl;
        errors++;
    }
    if(USE_ALTERNATIVE_SAMPLES){
        if ( ! vm["FilterType"].empty() ){
            cout << "  FilterType=" << vm["FilterType"].as<string>() << endl;
        }
        else{
            cout << "  [!] ERROR: Filter not set." << endl;
            errors++;
        }

        if(vm["KernelIdx"].defaulted() ){
            cout << "  KernelIdx not set. Using default value zero." << endl;
        }
        else{
            cout << "  KernelIdx=" << vm["KernelIdx"].as<int>() <<  endl;
        }
    }
    
    return errors;
}

int isFilterAvailable(string filterName){
    for(int idx=0; idx<availableFilters.size(); idx++){
        if( filterName == availableFilters[idx] )
            return 1;
    }
    return 0;
}

int isFilterAvailable_arm(string filterName){
    for(int idx=0; idx<availableFilters_arm.size(); idx++){
        if( filterName == availableFilters_arm[idx] )
            return 1;
    }
    return 0;
}

void print_timestamp(char* messagePreffix){
    gettimeofday(&tv, NULL);
    timeStruct = localtime(&tv.tv_sec);
    date_and_time.hour = timeStruct->tm_hour;
    date_and_time.minutes = timeStruct->tm_min;
    date_and_time.seconds = timeStruct->tm_sec;
    date_and_time.msec = (int) (tv.tv_usec / 1000);
                // hh:mm:ss:ms
    printf("%s @ %02d:%02d:%02d.%03d\n", messagePreffix,date_and_time.hour, date_and_time.minutes, date_and_time.seconds, date_and_time.msec );
}

// Save the start time (before write samples)
void save_startTime(){
    gettimeofday(&tv, NULL);
    timeStruct = localtime(&tv.tv_sec);
    startWriteSamples.hour = timeStruct->tm_hour;
    startWriteSamples.minutes = timeStruct->tm_min;
    startWriteSamples.seconds = timeStruct->tm_sec;
    startWriteSamples.msec = (int) (tv.tv_usec / 1000);
}

// Save the finish time (after read distortion)
void save_finishTime(){
    gettimeofday(&tv, NULL);
    timeStruct = localtime(&tv.tv_sec);
    endReadingDistortion.hour = timeStruct->tm_hour;
    endReadingDistortion.minutes = timeStruct->tm_min;
    endReadingDistortion.seconds = timeStruct->tm_sec;
    endReadingDistortion.msec = (int) (tv.tv_usec / 1000);

    timeDifference = computeTimeDifferenceMs(startWriteSamples, endReadingDistortion);
}

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

    // SizeId=0
    else if(cuSize==ALL_AL_4x4)
        return "ALL_AL_4x4";

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

void readMemobjsIntoArray_Distortion(cl_command_queue command_queue, int nCTUs, int nPredictionModes, cl_mem return_SAD_memObj, long *return_SAD, cl_mem return_SATD_memObj, long *return_SATD, cl_mem return_minSadHad_memObj, long *return_minSadHad, int currFrame){
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;

#if ! MAX_PERFORMANCE_DIST // When not HIGH PERFORMANCE, read SAD and SATD.
    error =  clEnqueueReadBuffer(command_queue, return_SAD_memObj, CL_TRUE, 0, 
            nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(cl_long), return_SAD, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, return_SATD_memObj, CL_TRUE, 0, 
            nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(cl_long), return_SATD, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;
#endif
    
    // ALWAYS read minSadhad

    error =  clEnqueueReadBuffer(command_queue, return_minSadHad_memObj, CL_FALSE, (currFrame%BUFFER_SLOTS) * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(cl_long), 
            // nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(cl_long), return_minSadHad + currFrame * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(long), 0, NULL, &read_event);
            nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(cl_long), return_minSadHad, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    //error = clWaitForEvents(1, &read_event);
    //probe_error(error, (char*)"Error waiting for read events\n");
    //error = clFinish(command_queue);
    //probe_error(error, (char*)"Error finishing read\n");
    //clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    //clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    //nanoSeconds += read_time_end-read_time_start;

    //readTime_SAD += nanoSeconds;
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

void reportAllDistortionValues_ALL(long int *SAD, long int *SATD, long int *minSadHad, int nCTUs){
    printf("-=-=-=-=-=-=-=-=- DISTORTION RESULTS FOR ALL CTUs -=-=-=-=-=-=-=-=-\n");
    printf("CTU,cuSize,CU,Mode,SAD,SATD,minSadHad\n");
    for(int ctu=0; ctu<nCTUs; ctu++){
        // SizeID=2
        for(int cuSize=0; cuSize<NUM_CU_SIZES_SizeId2; cuSize++){
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                    // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                    printf("%d,%s,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), cu, mode);  //  Report CU size/position info
                    printf("%ld,", SAD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                    printf("%ld,", SATD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                    printf("%ld\n", minSadHad[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                }
            }
        }
        // SizeID=1
        for(int offset=0; offset<NUM_CU_SIZES_SizeId1; offset++){
            int cuSize = offset+FIRST_SizeId1;
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                for(int mode=0; mode<PREDICTION_MODES_ID1*2; mode++){
                    // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                    printf("%d,%s,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), cu, mode);  //  Report CU size/position info
                    printf("%ld,", SAD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                    printf("%ld,", SATD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                    printf("%ld\n", minSadHad[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                }
            }
        }
        // SizeId=0
        for(int offset=0; offset<NUM_CU_SIZES_SizeId0; offset++){
            int cuSize = offset+FIRST_SizeId0;
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                for(int mode=0; mode<PREDICTION_MODES_ID0*2; mode++){
                    // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                    printf("%d,%s,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), cu, mode);  //  Report CU size/position info
                    printf("%ld,", SAD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                    printf("%ld,", SATD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                    printf("%ld\n", minSadHad[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                }
            }
        }
    }
}

void exportAllDistortionValues_File(long int *SAD, long int *SATD, long int *minSadHad, int nCTUs, int frameWidth, string outputFile){
    int ctuCols = ceil(frameWidth/128.0);

    FILE *distortionFile;
    string outputFileName = outputFile + ".csv";
    distortionFile = fopen(outputFileName.c_str(),"w");
    fprintf(distortionFile,"CTU,cuSizeName,W,H,CU,X,Y,Mode,SAD,SATD,minSadHad\n");

    // printf("-=-=-=-=-=-=-=-=- DISTORTION RESULTS FOR ALL CTUs -=-=-=-=-=-=-=-=-\n");
    // printf("CTU,cuSizeName,W,H,CU,X,Y,Mode,SAD,SATD\n");
    int ctuX, ctuY, cuX, cuY;
    for(int ctu=0; ctu<nCTUs; ctu++){
        ctuX = 128*(ctu%ctuCols);
        ctuY = 128*(ctu/ctuCols);
        
        // SizeID=2
        for(int cuSize=0; cuSize<NUM_CU_SIZES_SizeId2; cuSize++){
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                cuX = ctuX + ALL_X_POS[cuSize][cu];
                cuY = ctuY + ALL_Y_POS[cuSize][cu];

                for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                    fprintf(distortionFile, "%d,%s,%d,%d,%d,%d,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);
                    fprintf(distortionFile, "%ld,", SAD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                    fprintf(distortionFile, "%ld,", SATD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                    fprintf(distortionFile, "%ld\n", minSadHad[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                }
            }
        }

        // SizeID=1
        for(int offset=0; offset<NUM_CU_SIZES_SizeId1; offset++){
            int cuSize = offset+FIRST_SizeId1;
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                cuX = ctuX + ALL_X_POS[cuSize][cu];
                cuY = ctuY + ALL_Y_POS[cuSize][cu];

                for(int mode=0; mode<PREDICTION_MODES_ID1*2; mode++){
                    fprintf(distortionFile, "%d,%s,%d,%d,%d,%d,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);
                    fprintf(distortionFile, "%ld,", SAD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                    fprintf(distortionFile, "%ld,", SATD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                    fprintf(distortionFile, "%ld\n", minSadHad[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                }
            }
        }

        // SizeID=0
        for(int offset=0; offset<NUM_CU_SIZES_SizeId0; offset++){
            int cuSize = offset+FIRST_SizeId0;
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                cuX = ctuX + 4*(cu%32); // ALL_X_POS[cuSize][cu];
                cuY = ctuY + 4*(cu/32); // ALL_Y_POS[cuSize][cu];

                for(int mode=0; mode<PREDICTION_MODES_ID0*2; mode++){
                    fprintf(distortionFile, "%d,%s,%d,%d,%d,%d,%d,%d,", ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);
                    fprintf(distortionFile, "%ld,", SAD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                    fprintf(distortionFile, "%ld,", SATD[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                    fprintf(distortionFile, "%ld\n", minSadHad[ ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                }
            }
        }        
    }
    fclose(distortionFile);
}

void reportTargetDistortionValues_ALL(long int *SAD, long int *SATD, long int *minSadHad, int nCTUs, int targetCTU){
    printf("-=-=-=-=-=-=-=-=- DISTORTION RESULTS FOR CTU %d -=-=-=-=-=-=-=-=-\n", targetCTU);
    printf("CTU,cuSize,CU,Mode,SAD,SATD,minSadHad\n");
    // SizeID=2
    for(int cuSize=0; cuSize<NUM_CU_SIZES_SizeId2; cuSize++){
        for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
            for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                printf("%d,%s,%d,%d,", targetCTU, translateCuSizeIdx_ALL(cuSize), cu, mode);  //  Report CU size/position info
                printf("%ld,", SAD[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                printf("%ld,", SATD[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                printf("%ld\n", minSadHad[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
            }
        }
    }
    // SizeID=1
    for(int offset=0; offset<NUM_CU_SIZES_SizeId1; offset++){
        int cuSize = offset+FIRST_SizeId1;
        for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
            for(int mode=0; mode<PREDICTION_MODES_ID1*2; mode++){
                // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                printf("%d,%s,%d,%d,", targetCTU, translateCuSizeIdx_ALL(cuSize), cu, mode);  //  Report CU size/position info
                printf("%ld,", SAD[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                printf("%ld,", SATD[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                printf("%ld\n", minSadHad[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
            }
        }
    }
    // SizeID=0
    for(int offset=0; offset<NUM_CU_SIZES_SizeId0; offset++){
        int cuSize = offset+FIRST_SizeId0;
        for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
            for(int mode=0; mode<PREDICTION_MODES_ID0*2; mode++){
                // printf("%d,%d,%d,%d,", ctu, cuSize, cu, mode);  //  Report CU size/position info
                printf("%d,%s,%d,%d,", targetCTU, translateCuSizeIdx_ALL(cuSize), cu, mode);  //  Report CU size/position info
                printf("%ld,", SAD[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                printf("%ld,", SATD[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                printf("%ld\n", minSadHad[ targetCTU*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
            }
        }
    }
}

void reportTargetDistortionValues_File(long int *SAD, long int *SATD, long int *minSadHad, int targetCtu, int frameWidth, string outputFile, int nCTUs){
    int ctu = targetCtu;
    int ctuCols = ceil(frameWidth/128.0);

    FILE *distortionFile;
    string outputFileName = outputFile + ".csv";
    distortionFile = fopen(outputFileName.c_str(),"w");
    fprintf(distortionFile,"POC,CTU,cuSizeName,W,H,CU,X,Y,Mode,SAD,SATD,minSadHad\n");

    int ctuX, ctuY, cuX, cuY;
    ctuX = 128*(targetCtu%ctuCols);
    ctuY = 128*(targetCtu/ctuCols);

    for(int frame=0; frame<N_FRAMES; frame++){

        int frameStride = frame * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES];

        // SizeID=2
        for(int cuSize=0; cuSize<NUM_CU_SIZES_SizeId2; cuSize++){
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                cuX = ctuX + ALL_X_POS[cuSize][cu];
                cuY = ctuY + ALL_Y_POS[cuSize][cu];

                for(int mode=0; mode<PREDICTION_MODES_ID2*2; mode++){
                    fprintf(distortionFile, "%d,%d,%s,%d,%d,%d,%d,%d,%d,", frame, ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);
                    fprintf(distortionFile, "%ld,", SAD[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                    fprintf(distortionFile, "%ld,", SATD[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                    fprintf(distortionFile, "%ld\n", minSadHad[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID2*2 + mode ]);
                }
            }
        }
        // SizeID=1
        for(int offset=0; offset<NUM_CU_SIZES_SizeId1; offset++){
            int cuSize = offset+FIRST_SizeId1;
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                cuX = ctuX + ALL_X_POS[cuSize][cu];
                cuY = ctuY + ALL_Y_POS[cuSize][cu];

                for(int mode=0; mode<PREDICTION_MODES_ID1*2; mode++){
                    fprintf(distortionFile, "%d,%d,%s,%d,%d,%d,%d,%d,%d,", frame,ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);
                    fprintf(distortionFile, "%ld,", SAD[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                    fprintf(distortionFile, "%ld,", SATD[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                    fprintf(distortionFile, "%ld\n", minSadHad[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID1*2 + mode ]);
                }
            }
        }
        // SizeID=0
        for(int offset=0; offset<NUM_CU_SIZES_SizeId0; offset++){
            int cuSize = offset+FIRST_SizeId0;
            for(int cu=0; cu<ALL_cusPerCtu[cuSize]; cu++){
                cuX = ctuX + 4*(cu%32); //ALL_X_POS[cuSize][cu];
                cuY = ctuY + 4*(cu/32); // ALL_Y_POS[cuSize][cu];

                for(int mode=0; mode<PREDICTION_MODES_ID0*2; mode++){
                    fprintf(distortionFile, "%d,%d,%s,%d,%d,%d,%d,%d,%d,", frame,ctu, translateCuSizeIdx_ALL(cuSize), ALL_widths[cuSize], ALL_heights[cuSize], cu, cuX, cuY, mode);
                    fprintf(distortionFile, "%ld,", SAD[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                    fprintf(distortionFile, "%ld,", SATD[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                    fprintf(distortionFile, "%ld\n", minSadHad[ frameStride + ctu*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] + ALL_stridedDistortionsPerCtu[cuSize] + cu*PREDICTION_MODES_ID0*2 + mode ]);
                }
            }
        }    
    }
    fclose(distortionFile);
}

void reportTimingResults_Compact(){
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    printf("TIMING RESULTS (miliseconds)\n");
    printf("Elapsed time (ms) from writing samples to reading distortion (%dx), %d\n", N_FRAMES, timeDifference);// + 1000*timeDifference.seconds + 1000*60*timeDifference.minutes);
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n");

}


void reportTimingResults(){
    totalGpuTime = writeTime + execTime_reducedBoundaries + execTime_reducedPrediction + execTime_upsampleDistortion + readTime_SAD;

    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n");
    printf("TIMING RESULTS (nanoseconds)\n");
    printf("Write(%dx),%f,%.3f\n", N_FRAMES, writeTime, writeTime/totalGpuTime);
    printf("InitBoundaries(%dx):  Execution,%f,%.3f\n", N_FRAMES, execTime_reducedBoundaries, execTime_reducedBoundaries/totalGpuTime);
    if(!TRACE_POWER){
        printf("InitBoundaries(%dx):  Read size-reduced,%f\n", N_FRAMES, readTime_reducedBoundariesSize);
        printf("InitBoundaries(%dx):  Read size-complete,%f\n", N_FRAMES, readTime_completeBoundariesSize);
        printf("InitBoundaries(%dx):  Read unified-reduced,%f\n", N_FRAMES, readTime_reducedBoundariesUnified);
        printf("InitBoundaries(%dx):  Read unified-complete,%f\n", N_FRAMES, readTime_completeBoundariesUnified);
    }
    printf("ReducedPrediction(%dx):  Execution,%f,%.3f\n", N_FRAMES, execTime_reducedPrediction, execTime_reducedPrediction/totalGpuTime);
    if(!TRACE_POWER)
        printf("ReducedPrediction(%dx):  Read,%f\n", N_FRAMES, readTime_reducedPrediction);   

    printf("UpsamplePredictionDistortion Id=2(%dx): Execution,%f,%.3f\n", N_FRAMES, execTime_upsampleDistortion_SizeId2, execTime_upsampleDistortion_SizeId2/totalGpuTime); 
    printf("UpsamplePredictionDistortion Id=1(%dx): Execution,%f,%.3f\n", N_FRAMES, execTime_upsampleDistortion_SizeId1, execTime_upsampleDistortion_SizeId1/totalGpuTime); 
    printf("UpsamplePredictionDistortion Id=0(%dx): Execution,%f,%.3f\n", N_FRAMES, execTime_upsampleDistortion_SizeId0, execTime_upsampleDistortion_SizeId0/totalGpuTime); 
    printf("UpsamplePredictionDistortion(%dx): Execution,%f,%.3f\n", N_FRAMES, execTime_upsampleDistortion, execTime_upsampleDistortion/totalGpuTime); 
    printf("UpsamplePredictionDistortion(%dx): Read,%f,%.3f\n", N_FRAMES, readTime_SAD, readTime_SAD/totalGpuTime); 
    printf("W_E_R(%dx):TotalGpuTime, %f\n", N_FRAMES, totalGpuTime); 
    printf("=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=\n\n");    
}


void readMemobjsIntoArray_UnifiedBoundaries(cl_command_queue command_queue, int nCTUs, cl_mem redT_all_memObj, cl_mem refT_all_memObj, short *return_unified_redT, short *return_unified_refT, cl_mem redL_all_memObj, cl_mem refL_all_memObj, short *return_unified_redL, short *return_unified_refL){
    
    int error;
    double nanoSeconds = 0.0;
    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    
    error =  clEnqueueReadBuffer(command_queue, redT_all_memObj, CL_TRUE, 0, 
            nCTUs * (ALL_TOTAL_CUS_SizeId12_PER_CTU * BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU * BOUNDARY_SIZE_Id0) * sizeof(cl_short), return_unified_redT, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return reduced boundaries\n");
    error = clWaitForEvents(1, &read_event);
    probe_error(error, (char*)"Error waiting for read events\n");
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing read\n");
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds += read_time_end-read_time_start;

    error =  clEnqueueReadBuffer(command_queue, redL_all_memObj, CL_TRUE, 0, 
            nCTUs * (ALL_TOTAL_CUS_SizeId12_PER_CTU * BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU * BOUNDARY_SIZE_Id0) * sizeof(cl_short), return_unified_redL, 0, NULL, &read_event);
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
    int ctuIdx = targetCTU * (ALL_TOTAL_CUS_SizeId12_PER_CTU * BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU * BOUNDARY_SIZE_Id0);

    printf("=-=-=-=-=- REDUCED TOP BOUNDARIES RESULTS -=-=-=-=-=\n");
    for(int cuSizeIdx=0; cuSizeIdx<ALL_NUM_CU_SIZES; cuSizeIdx++){
        int boundarySize = ALL_reducedBoundarySizes[cuSizeIdx];
        printf("RESULTS FOR %s\n", translateCuSizeIdx_ALL(cuSizeIdx));
        for (int cu = 0; cu < ALL_cusPerCtu[cuSizeIdx]; cu++){
            printf("CU %d\n", cu);
            for(int b=0; b<boundarySize; b++){
                // Even though CUs 4x4 have reducedBoundarySize=2, the ALL_stridedCusPerCtu points ot the start of the current CU size and all previous sizes have reducedboundarySize=4
                printf("%d,", unified_redT[ctuIdx + ALL_stridedCusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY + cu*boundarySize + b]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("=-=-=-=-=- REDUCED LEFT BOUNDARIES RESULTS -=-=-=-=-=\n");
    for(int cuSizeIdx=0; cuSizeIdx<ALL_NUM_CU_SIZES; cuSizeIdx++){
        int boundarySize = ALL_reducedBoundarySizes[cuSizeIdx];
        printf("RESULTS FOR %s\n", translateCuSizeIdx_ALL(cuSizeIdx));
        for (int cu = 0; cu < ALL_cusPerCtu[cuSizeIdx]; cu++){
            printf("CU %d\n", cu);
            for(int b=0; b<boundarySize; b++){
                // Even though CUs 4x4 have reducedBoundarySize=2, the ALL_stridedCusPerCtu points ot the start of the current CU size and all previous sizes have reducedboundarySize=4
                printf("%d,", unified_redL[ctuIdx + ALL_stridedCusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY + cu*boundarySize + b]);
            }
            printf("\n");
        }
        printf("\n");
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

void readMemobjsIntoArray_FilteredFrame(cl_command_queue command_queue_common, int frameWidth, int frameHeight, cl_mem filteredFrame_memObj, short *return_filteredFrame){

    cl_ulong read_time_start, read_time_end;
    cl_event read_event;
    double nanoSeconds;

    int error =  clEnqueueReadBuffer(command_queue_common, filteredFrame_memObj, CL_TRUE, 0, 
            frameWidth*frameHeight * sizeof(cl_short), return_filteredFrame, 0, NULL, &read_event);
    probe_error(error, (char*)"Error reading return prediction\n");
    error = clFinish(command_queue_common);
    probe_error(error, (char*)"Error finishing read\n");


    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(read_time_start), &read_time_start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(read_time_end), &read_time_end, NULL);
    nanoSeconds = read_time_end-read_time_start;

    readTime_filter = nanoSeconds;
    
}

void correctFilteringAtEdges(int frameWidth, int frameHeight, unsigned short *reference_frame, short *return_filteredFrame){
    int y, x;

    // Top edge
    y=0;
    for(x=0; x<frameWidth; x++){
        return_filteredFrame[y*frameWidth + x] = reference_frame[y*frameWidth + x];
    }
    
    // Bottom edge
    y=frameHeight-1;
    for(x=0; x<frameWidth; x++){
        return_filteredFrame[y*frameWidth + x] = reference_frame[y*frameWidth + x];
    }

    // Left edge
    x = 0;
    for(y=1; y<frameHeight-1; y++){
        return_filteredFrame[y*frameWidth + x] = reference_frame[y*frameWidth + x];
    }

    // Right edge
    x = frameWidth-1;
    for(y=1; y<frameHeight-1; y++){
        return_filteredFrame[y*frameWidth + x] = reference_frame[y*frameWidth + x];
    }
}

void optFilterCpuInt(unsigned short *input, unsigned short *output, int width, int height, int kernelIdx){

    unsigned short kernel[3][3];
    
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            kernel[i][j] = convKernelLib[kernelIdx][i][j];
        }            
    }

    // printf(">> INITIALIZED THE COEFFS\n");

    int fullScale = 0;   for(int i=0; i<3; i++) for(int j=0; j<3; j++) fullScale += kernel[i][j];
    int edgeScale = 0;   for(int i=0; i<2; i++) for(int j=0; j<3; j++) edgeScale += kernel[i][j];
    int cornerScale = 0; for(int i=0; i<2; i++) for(int j=0; j<2; j++) cornerScale += kernel[i][j];
    int result = 0;
    int currScale = 0;

    int h,w,i,j;
    for(int f=0; f<MULTIPLIER_CPU_FILTER; f++){
        for(h=1; h<height-1; h++){
            for(w=1; w<width-1; w++){
                
                currScale = fullScale;

                for(i=-1; i<=1; i++){
                    for(j=-1; j<=1; j++){
                        result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                    }
                }

                output[h*width+w] = (result + currScale/2)/currScale;
                result = 0.0;
            }
        }

        // printf(">> FILTERED CENTER\n");


        // First line, except corners. Without top coeffs
        h=0;
        currScale = edgeScale;
        for(w=1; w<width-1; w++){
            result = 0;
            for(i=0; i<=1; i++){
                for(j=-1; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }      
            output[h*width+w] = (result + currScale/2)/currScale;  
        }

        // printf(">> FILTERED TOP\n");

        // Last line, except corners. Without bottom coeffs
        h=height-1;
        currScale = edgeScale;
        for(w=1; w<width-1; w++){
            result = 0;
            for(i=-1; i<=0; i++){
                for(j=-1; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }        
            output[h*width+w] = (result + currScale/2)/currScale;
        }    

        // printf(">> FILTERED bottom\n");

        // First column, except corners. Without left coeffs
        w=0;
        currScale = edgeScale;
        for(h=1; h<height-1; h++){
            result = 0;
            for(i=-1; i<=1; i++){
                for(j=0; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }  
            output[h*width+w] = (result + currScale/2)/currScale;      
        }

        // printf(">> FILTERED LEFT\n");

        // Last column, except corners. Without right coeffs
        w=width-1;
        currScale = edgeScale;
        for(h=1; h<height-1; h++){
            result = 0;
            for(i=-1; i<=1; i++){
                for(j=-1; j<=0; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }   
            output[h*width+w] = (result + currScale/2)/currScale;     
        }

        // printf(">> FILTERED RIGHT\n");

        // Corners
        // Top-left
        result=0; w=0; h=0; currScale=cornerScale;
        for(i=0; i<=1; i++)
            for(j=0; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = (result + currScale/2)/currScale;     


        // Top-right
        result=0; w=width-1; h=0; currScale=cornerScale;
        for(i=0; i<=1; i++)
            for(j=-1; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Bottom-left
        result=0; w=0; h=height-1; currScale=cornerScale;
        for(i=-1; i<=0; i++)
            for(j=0; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Top-left
        result=0; w=width-1; h=height-1; currScale=cornerScale;
        for(i=-1; i<=0; i++)
            for(j=-1; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = (result + currScale/2)/currScale;         

        // printf(">> FILTERED EDGES\n");

    }    

}

int getNumCtus(int frameWidth, int frameHeight){
    pair<int,int> inputRes(frameWidth, frameHeight);

    for( unsigned int i=0; i<availableRes.size(); i++){
        if(inputRes == pair<int,int>( get<0>(availableRes[i]), get<1>(availableRes[i]) )){
            return get<2>(availableRes[i]); 
        }
    }
    // In case it didn't match with any of the available resolutions, return an error code
   return 0;
}


void parallelOptFilterCpuInt_3x3(unsigned short *input, unsigned short *output, int width, int height, int kernelIdx, int nThreads){

    int kernel[3][3];
    
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            kernel[i][j] = convKernelLib[kernelIdx][i][j];
        }
            
    }


    int fullScale = 0;   for(int i=0; i<3; i++) for(int j=0; j<3; j++) fullScale += kernel[i][j];
    int edgeScale = 0;   for(int i=0; i<2; i++) for(int j=0; j<3; j++) edgeScale += kernel[i][j];
    int cornerScale = 0; for(int i=0; i<2; i++) for(int j=0; j<2; j++) cornerScale += kernel[i][j];
    int result = 0;
    int currScale = 0;

    int h,w,i,j;
    for(int f=0; f<MULTIPLIER_CPU_FILTER; f++){
        currScale = fullScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=1; h<height-1; h++){
            for(w=1; w<width-1; w++){
                result = 0.0;
                for(i=-1; i<=1; i++){
                    for(j=-1; j<=1; j++){
                        result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                    }
                }
                output[h*width+w] = (result + currScale/2)/currScale;
            }
        }

        // First or last lines, except corners. Without top or bottom coeffs
        currScale = edgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(w=1; w<width-1; w++){
            // Top
            h=0;
            result = 0;
            for(i=0; i<=1; i++){
                for(j=-1; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }      
            output[h*width+w] = (result + currScale/2)/currScale;  

            // Bottom
            h=height-1;
            result = 0;
            for(i=-1; i<=0; i++){
                for(j=-1; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }      
            output[h*width+w] = (result + currScale/2)/currScale;  
        }

        // // Last line, except corners. Without bottom coeffs
        // h=height-1;
        // currScale = edgeScale;
        // #pragma omp parallel for private(result, w, i, j) num_threads(nThreads)
        // for(w=1; w<width-1; w++){
        //     result = 0;
        //     for(i=-1; i<=0; i++){
        //         for(j=-1; j<=1; j++){
        //             result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
        //         }
        //     }        
        //     output[h*width+w] = (result + currScale/2)/currScale;
        // }    

        // First or last column, except corners. Without left or right coeffs
        currScale = edgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=1; h<height-1; h++){
            // Left
            w=0;
            result = 0;
            for(i=-1; i<=1; i++){
                for(j=0; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }  
            output[h*width+w] = (result + currScale/2)/currScale;      

            // Right
            w=width-1;
            result = 0;
            for(i=-1; i<=1; i++){
                for(j=-1; j<=0; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }  
            output[h*width+w] = (result + currScale/2)/currScale;      
        }

        // // Last column, except corners. Without right coeffs
        // w=width-1;
        // currScale = edgeScale;
        // #pragma omp parallel for private(result, h, i, j) num_threads(nThreads)
        // for(h=1; h<height-1; h++){
        //     result = 0;
        //     for(i=-1; i<=1; i++){
        //         for(j=-1; j<=0; j++){
        //             result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
        //         }
        //     }   
        //     output[h*width+w] = (result + currScale/2)/currScale;     
        // }

        // Corners
        // Top-left
        result=0; w=0; h=0; currScale=cornerScale;
        for(i=0; i<=1; i++)
            for(j=0; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = (result + currScale/2)/currScale;     


        // Top-right
        result=0; w=width-1; h=0; currScale=cornerScale;
        for(i=0; i<=1; i++)
            for(j=-1; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Bottom-left
        result=0; w=0; h=height-1; currScale=cornerScale;
        for(i=-1; i<=0; i++)
            for(j=0; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Top-left
        result=0; w=width-1; h=height-1; currScale=cornerScale;
        for(i=-1; i<=0; i++)
            for(j=-1; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = (result + currScale/2)/currScale;         

    }    

}

void parallelOptFilterCpuInt_5x5(unsigned short *input, unsigned short *output, int width, int height, int kernelIdx, int nThreads){

    int kernel[5][5];
    
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            kernel[i][j] = convKernelLib_5x5[kernelIdx][i][j];
        }
    }

    // printf("Kernel:\n");
    // for(int i=0; i<5; i++){
    //     for(int j=0; j<5; j++){
    //         printf("%d,", kernel[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    int fullScale = 0;          for(int i=0; i<5; i++) for(int j=0; j<5; j++) fullScale += kernel[i][j];
    // printf("FullScale: %d\n", fullScale);
    
    int outerCornerScale = 0;   for(int i=0; i<3; i++) for(int j=0; j<3; j++) outerCornerScale += kernel[i][j];
    int innerCornerScale = 0;   for(int i=0; i<4; i++) for(int j=0; j<4; j++) innerCornerScale += kernel[i][j];

    int outerEdgeScale = 0;     for(int i=0; i<3; i++) for(int j=0; j<5; j++) outerEdgeScale += kernel[i][j];
    int innerEdgeScale = 0;     for(int i=0; i<4; i++) for(int j=0; j<5; j++) innerEdgeScale += kernel[i][j];

    int interfaceScale = 0;     for(int i=0; i<4; i++) for(int j=0; j<3; j++) interfaceScale += kernel[i][j];


    
    int result = 0;
    int currScale = 0;

    int h,w,i,j;
    for(int f=0; f<MULTIPLIER_CPU_FILTER; f++){

        currScale = fullScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=2; h<height-2; h++){
            for(w=2; w<width-2; w++){
                result = 0.0;
                // if(w==2 && h==2){
                //     printf("Result @(2x2) -> ");
                // }
                for(i=-2; i<=2; i++){
                    for(j=-2; j<=2; j++){
                        result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                        // if(w==2 && h==2)
                        //     printf("%d, ", result);
                    }
                }
                // if(w==2 && h==2)
                //     printf(" -> (%d+%d)/%d = %d\n", result, currScale/2, currScale, (result + currScale/2)/currScale);

                output[h*width+w] = (result + currScale/2)/currScale;
            }
        }

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Outer Edges

        // First or last lines, except corners. Without top or bottom coeffs
        currScale = outerEdgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(w=2; w<width-2; w++){
            // Top
            h=0;
            result = 0;
            for(i=0; i<=2; i++){
                for(j=-2; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }      
            output[h*width+w] = (result + currScale/2)/currScale;  

            // Bottom
            h=height-1;
            result = 0;
            for(i=-2; i<=0; i++){
                for(j=-2; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }      
            output[h*width+w] = (result + currScale/2)/currScale;  
        }

   

        // First or last column, except corners. Without left or right coeffs
        currScale = outerEdgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=2; h<height-2; h++){
            // Left
            w=0;
            result = 0;
            for(i=-2; i<=2; i++){
                for(j=0; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }  
            output[h*width+w] = (result + currScale/2)/currScale;      

            // Right
            w=width-1;
            result = 0;
            for(i=-2; i<=2; i++){
                for(j=-2; j<=0; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }  
            output[h*width+w] = (result + currScale/2)/currScale;      
        }


        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Inner Edges


        // Second or pre-last lines, except corners. Without top or bottom coeffs
        currScale = innerEdgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(w=2; w<width-2; w++){
            // Top
            h=1;
            result = 0;
            for(i=-1; i<=2; i++){
                for(j=-2; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }      
            output[h*width+w] = (result + currScale/2)/currScale;  

            // Bottom
            h=height-2;
            result = 0;
            for(i=-2; i<=1; i++){
                for(j=-2; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }      
            output[h*width+w] = (result + currScale/2)/currScale;  
        }

   

        // Second or pre-last column, except corners. Without left or right coeffs
        currScale = innerEdgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=2; h<height-2; h++){
            // Left
            w=1;
            result = 0;
            for(i=-2; i<=2; i++){
                for(j=-1; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }  
            output[h*width+w] = (result + currScale/2)/currScale;      

            // Right
            w=width-2;
            result = 0;
            for(i=-2; i<=2; i++){
                for(j=-2; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }  
            output[h*width+w] = (result + currScale/2)/currScale;      
        }


        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Outer corners

        // Top-left
        result=0; w=0; h=0; currScale=outerCornerScale;
        for(i=0; i<=2; i++)
            for(j=0; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     


        // Top-right
        result=0; w=width-1; h=0; currScale=outerCornerScale;
        for(i=0; i<=2; i++)
            for(j=-2; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Bottom-left
        result=0; w=0; h=height-1; currScale=outerCornerScale;
        for(i=-2; i<=0; i++)
            for(j=0; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Top-left
        result=0; w=width-1; h=height-1; currScale=outerCornerScale;
        for(i=-2; i<=0; i++)
            for(j=-2; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;         


        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Inner corners

        // Top-left
        result=0; w=1; h=1; currScale=innerCornerScale;
        for(i=-1; i<=2; i++)
            for(j=-1; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     


        // Top-right
        result=0; w=width-2; h=1; currScale=innerCornerScale;
        for(i=-1; i<=2; i++)
            for(j=-2; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Bottom-left
        result=0; w=1; h=height-2; currScale=innerCornerScale;
        for(i=-2; i<=1; i++)
            for(j=-1; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Top-left
        result=0; w=width-2; h=height-2; currScale=innerCornerScale;
        for(i=-2; i<=1; i++)
            for(j=-2; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;         



        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Interface


        // Top-left
        result=0; w=1; h=0; currScale=interfaceScale;
        for(i=0; i<=2; i++)
            for(j=-1; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        result=0; w=0; h=1; currScale=interfaceScale;
        for(i=-1; i<=2; i++)
            for(j=0; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Top-right
        result=0; w=width-2; h=0; currScale=interfaceScale;
        for(i=0; i<=2; i++)
            for(j=-2; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        result=0; w=width-1; h=1; currScale=interfaceScale;
        for(i=-1; i<=2; i++)
            for(j=-2; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     


        // Bottom-left
        result=0; w=0; h=height-2; currScale=interfaceScale;
        for(i=-2; i<=1; i++)
            for(j=0; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        result=0; w=1; h=height-1; currScale=interfaceScale;
        for(i=-2; i<=0; i++)
            for(j=-1; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;     

        // Bottom-right
        result=0; w=width-2; h=height-1; currScale=interfaceScale;
        for(i=-2; i<=0; i++)
            for(j=-2; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;  

        result=0; w=width-1; h=height-2; currScale=interfaceScale;
        for(i=-2; i<=1; i++)
            for(j=-2; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = (result + currScale/2)/currScale;  

    }  

}

void parallelOptFilterCpuFloat_3x3(unsigned short *input, unsigned short *output, int width, int height, int kernelIdx, int nThreads){

    double kernel[3][3];
    
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            kernel[i][j] = convKernelLib_float[kernelIdx][i][j];
        }
            
    }


    double fullScale = 0;   for(int i=0; i<3; i++) for(int j=0; j<3; j++) fullScale += kernel[i][j];
    double edgeScale = 0;   for(int i=0; i<2; i++) for(int j=0; j<3; j++) edgeScale += kernel[i][j];
    double cornerScale = 0; for(int i=0; i<2; i++) for(int j=0; j<2; j++) cornerScale += kernel[i][j];
    double result = 0;
    double currScale = 0;

    int h,w,i,j;
    for(int f=0; f<MULTIPLIER_CPU_FILTER; f++){
        currScale = fullScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=1; h<height-1; h++){
            for(w=1; w<width-1; w++){
                result = 0.0;
                // if(h==1 && w==1)
                //     printf("Result: ");
                for(i=-1; i<=1; i++){
                    for(j=-1; j<=1; j++){
                        result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                        // if(h==1 && w==1)
                        //     printf("%f,: ", result);
                    }
                }
                // if(h==1 && w==1)
                //     printf("\nFinal result: (%f/%f)=%f\n,: ", result, currScale, round(result/currScale));
                output[h*width+w] = round(result/currScale);
            }
        }

        // First or last lines, except corners. Without top or bottom coeffs
        currScale = edgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(w=1; w<width-1; w++){
            // Top
            h=0;
            result = 0;
            for(i=0; i<=1; i++){
                for(j=-1; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }      
            output[h*width+w] = round(result/currScale);

            // Bottom
            h=height-1;
            result = 0;
            for(i=-1; i<=0; i++){
                for(j=-1; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }      
            output[h*width+w] = round(result/currScale);
        }

        // // Last line, except corners. Without bottom coeffs
        // h=height-1;
        // currScale = edgeScale;
        // #pragma omp parallel for private(result, w, i, j) num_threads(nThreads)
        // for(w=1; w<width-1; w++){
        //     result = 0;
        //     for(i=-1; i<=0; i++){
        //         for(j=-1; j<=1; j++){
        //             result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
        //         }
        //     }        
        //     output[h*width+w] = (result + currScale/2)/currScale;
        // }    

        // First or last column, except corners. Without left or right coeffs
        currScale = edgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=1; h<height-1; h++){
            // Left
            w=0;
            result = 0;
            for(i=-1; i<=1; i++){
                for(j=0; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }  
            output[h*width+w] = (result + currScale/2)/currScale;      

            // Right
            w=width-1;
            result = 0;
            for(i=-1; i<=1; i++){
                for(j=-1; j<=0; j++){
                    result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
                }
            }  
            output[h*width+w] = round(result/currScale);
        }

        // // Last column, except corners. Without right coeffs
        // w=width-1;
        // currScale = edgeScale;
        // #pragma omp parallel for private(result, h, i, j) num_threads(nThreads)
        // for(h=1; h<height-1; h++){
        //     result = 0;
        //     for(i=-1; i<=1; i++){
        //         for(j=-1; j<=0; j++){
        //             result += input[(h+i)*width + w+j]*kernel[1+i][1+j];    
        //         }
        //     }   
        //     output[h*width+w] = (result + currScale/2)/currScale;     
        // }

        // Corners
        // Top-left
        result=0; w=0; h=0; currScale=cornerScale;
        for(i=0; i<=1; i++)
            for(j=0; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = round(result/currScale);


        // Top-right
        result=0; w=width-1; h=0; currScale=cornerScale;
        for(i=0; i<=1; i++)
            for(j=-1; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = round(result/currScale);

        // Bottom-left
        result=0; w=0; h=height-1; currScale=cornerScale;
        for(i=-1; i<=0; i++)
            for(j=0; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = round(result/currScale);

        // Top-left
        result=0; w=width-1; h=height-1; currScale=cornerScale;
        for(i=-1; i<=0; i++)
            for(j=-1; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[1+i][1+j];
        output[h*width+w] = round(result/currScale);

    }    

}


void parallelOptFilterCpuFloat_5x5(unsigned short *input, unsigned short *output, int width, int height, int kernelIdx, int nThreads){

    double kernel[5][5];
    
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            kernel[i][j] = convKernelLib_5x5_float[kernelIdx][i][j];
        }
    }

    // printf("Kernel:\n");
    // for(int i=0; i<5; i++){
    //     for(int j=0; j<5; j++){
    //         printf("%d,", kernel[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    double fullScale = 0;          for(int i=0; i<5; i++) for(int j=0; j<5; j++) fullScale += kernel[i][j];
    // printf("FullScale: %d\n", fullScale);
    
    double outerCornerScale = 0;   for(int i=0; i<3; i++) for(int j=0; j<3; j++) outerCornerScale += kernel[i][j];
    double innerCornerScale = 0;   for(int i=0; i<4; i++) for(int j=0; j<4; j++) innerCornerScale += kernel[i][j];

    double outerEdgeScale = 0;     for(int i=0; i<3; i++) for(int j=0; j<5; j++) outerEdgeScale += kernel[i][j];
    double innerEdgeScale = 0;     for(int i=0; i<4; i++) for(int j=0; j<5; j++) innerEdgeScale += kernel[i][j];

    double interfaceScale = 0;     for(int i=0; i<4; i++) for(int j=0; j<3; j++) interfaceScale += kernel[i][j];


    
    double result = 0;
    double currScale = 0;

    int h,w,i,j;
    for(int f=0; f<MULTIPLIER_CPU_FILTER; f++){

        currScale = fullScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=2; h<height-2; h++){
            for(w=2; w<width-2; w++){
                result = 0.0;
                // if(w==2 && h==2){
                //     printf("Result @(2x2) -> ");
                // }
                for(i=-2; i<=2; i++){
                    for(j=-2; j<=2; j++){
                        result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                        // if(w==2 && h==2)
                        //     printf("%d, ", result);
                    }
                }
                // if(w==2 && h==2)
                //     printf(" -> (%d+%d)/%d = %d\n", result, currScale/2, currScale, (result + currScale/2)/currScale);

                output[h*width+w] = round(result/currScale);
            }
        }

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Outer Edges

        // First or last lines, except corners. Without top or bottom coeffs
        currScale = outerEdgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(w=2; w<width-2; w++){
            // Top
            h=0;
            result = 0;
            for(i=0; i<=2; i++){
                for(j=-2; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }      
            output[h*width+w] = round(result/currScale);

            // Bottom
            h=height-1;
            result = 0;
            for(i=-2; i<=0; i++){
                for(j=-2; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }      
            output[h*width+w] = round(result/currScale);
        }

   

        // First or last column, except corners. Without left or right coeffs
        currScale = outerEdgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=2; h<height-2; h++){
            // Left
            w=0;
            result = 0;
            for(i=-2; i<=2; i++){
                for(j=0; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }  
            output[h*width+w] = round(result/currScale);

            // Right
            w=width-1;
            result = 0;
            for(i=-2; i<=2; i++){
                for(j=-2; j<=0; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }  
            output[h*width+w] = round(result/currScale);
        }


        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Inner Edges


        // Second or pre-last lines, except corners. Without top or bottom coeffs
        currScale = innerEdgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(w=2; w<width-2; w++){
            // Top
            h=1;
            result = 0;
            for(i=-1; i<=2; i++){
                for(j=-2; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }      
            output[h*width+w] = round(result/currScale);

            // Bottom
            h=height-2;
            result = 0;
            for(i=-2; i<=1; i++){
                for(j=-2; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }      
            output[h*width+w] = round(result/currScale);
        }

   

        // Second or pre-last column, except corners. Without left or right coeffs
        currScale = innerEdgeScale;
        #pragma omp parallel for private(result, h, w, i, j) num_threads(nThreads)
        for(h=2; h<height-2; h++){
            // Left
            w=1;
            result = 0;
            for(i=-2; i<=2; i++){
                for(j=-1; j<=2; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }  
            output[h*width+w] = round(result/currScale); 

            // Right
            w=width-2;
            result = 0;
            for(i=-2; i<=2; i++){
                for(j=-2; j<=1; j++){
                    result += input[(h+i)*width + w+j]*kernel[2+i][2+j];    
                }
            }  
            output[h*width+w] = round(result/currScale);
        }


        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Outer corners

        // Top-left
        result=0; w=0; h=0; currScale=outerCornerScale;
        for(i=0; i<=2; i++)
            for(j=0; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);


        // Top-right
        result=0; w=width-1; h=0; currScale=outerCornerScale;
        for(i=0; i<=2; i++)
            for(j=-2; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        // Bottom-left
        result=0; w=0; h=height-1; currScale=outerCornerScale;
        for(i=-2; i<=0; i++)
            for(j=0; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        // Top-left
        result=0; w=width-1; h=height-1; currScale=outerCornerScale;
        for(i=-2; i<=0; i++)
            for(j=-2; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);


        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Inner corners

        // Top-left
        result=0; w=1; h=1; currScale=innerCornerScale;
        for(i=-1; i<=2; i++)
            for(j=-1; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);


        // Top-right
        result=0; w=width-2; h=1; currScale=innerCornerScale;
        for(i=-1; i<=2; i++)
            for(j=-2; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        // Bottom-left
        result=0; w=1; h=height-2; currScale=innerCornerScale;
        for(i=-2; i<=1; i++)
            for(j=-1; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        // Top-left
        result=0; w=width-2; h=height-2; currScale=innerCornerScale;
        for(i=-2; i<=1; i++)
            for(j=-2; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);



        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        //
        //        Interface


        // Top-left
        result=0; w=1; h=0; currScale=interfaceScale;
        for(i=0; i<=2; i++)
            for(j=-1; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        result=0; w=0; h=1; currScale=interfaceScale;
        for(i=-1; i<=2; i++)
            for(j=0; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        // Top-right
        result=0; w=width-2; h=0; currScale=interfaceScale;
        for(i=0; i<=2; i++)
            for(j=-2; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        result=0; w=width-1; h=1; currScale=interfaceScale;
        for(i=-1; i<=2; i++)
            for(j=-2; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        // Bottom-left
        result=0; w=0; h=height-2; currScale=interfaceScale;
        for(i=-2; i<=1; i++)
            for(j=0; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        result=0; w=1; h=height-1; currScale=interfaceScale;
        for(i=-2; i<=0; i++)
            for(j=-1; j<=2; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        // Bottom-right
        result=0; w=width-2; h=height-1; currScale=interfaceScale;
        for(i=-2; i<=0; i++)
            for(j=-2; j<=1; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

        result=0; w=width-1; h=height-2; currScale=interfaceScale;
        for(i=-2; i<=1; i++)
            for(j=-2; j<=0; j++)
                result += input[(h+i)*width + w+j]*kernel[2+i][2+j];
        output[h*width+w] = round(result/currScale);

    }  

}

void profileCpuFiltering(unsigned short *input, int frameWidth, int frameHeight, int kernelDim, int kernelIdx, int maxThreads, int reportFilterResults){
    printf("profileCpuFiltering\n");
    
    unsigned short *cpu_filtered_frame_parallel_int = (unsigned short *)malloc(sizeof(short) * frameWidth*frameHeight * N_FRAMES);
    unsigned short *cpu_filtered_frame_parallel_float = (unsigned short *)malloc(sizeof(short) * frameWidth*frameHeight * N_FRAMES);
    
    struct timeval tvCpu;
    struct tm *tmCpu;
    DateAndTime startFilter, finishFilter; // Used to track the processing time

    gettimeofday(&tvCpu, NULL);  
    tmCpu = localtime(&tvCpu.tv_sec);
    startFilter.hour = tmCpu->tm_hour;
    startFilter.minutes = tmCpu->tm_min;
    startFilter.seconds = tmCpu->tm_sec;
    startFilter.msec = (int) (tvCpu.tv_usec / 1000);

   // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
   //
   // FILTERING WITH INTEGER COEFFICIENTS
       
    if(kernelDim==3){
        print_timestamp((char*) "START CPU FILTER OPT INT 3x3 PARALLEL");    
    }
    else if(kernelDim==5){
        print_timestamp((char*) "START CPU FILTER OPT INT 5x5 PARALLEL");     
    }
    else{
        printf("ERROR: kernelDim=%d\n", kernelDim);
        return;
    }
    

    for(int t=1; t<=maxThreads; t++){
        // omp_set_num_threads(t);
        
        gettimeofday(&tvCpu, NULL);  
        tmCpu = localtime(&tvCpu.tv_sec);
        startFilter.hour = tmCpu->tm_hour;
        startFilter.minutes = tmCpu->tm_min;
        startFilter.seconds = tmCpu->tm_sec;
        startFilter.msec = (int) (tvCpu.tv_usec / 1000);

        if(kernelDim==3){
            parallelOptFilterCpuInt_3x3(input, cpu_filtered_frame_parallel_int, frameWidth, frameHeight, kernelIdx, t);
        }
        else{
            parallelOptFilterCpuInt_5x5(input, cpu_filtered_frame_parallel_int, frameWidth, frameHeight, kernelIdx, t);
        }
        

        gettimeofday(&tvCpu, NULL);
        tmCpu = localtime(&tvCpu.tv_sec);
        finishFilter.hour = tmCpu->tm_hour;
        finishFilter.minutes = tmCpu->tm_min;
        finishFilter.seconds = tmCpu->tm_sec;
        finishFilter.msec = (int) (tvCpu.tv_usec / 1000);
        timeDifference = computeTimeDifferenceMs(startFilter, finishFilter);

        printf(">> PARALLEL [%d threads] CPU FILTER [INT] %dx%d took %d ms\n", t, kernelDim, kernelDim, timeDifference);        
        
    }

    if(kernelDim==3){
        print_timestamp((char*) "END CPU FILTER OPT INT 3x3 PARALLEL");          
    }
    else{
        print_timestamp((char*) "END CPU FILTER OPT INT 5x5 PARALLEL");          
    }

    
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    // FILTERING WITH FLOATING-POINT COEFFICIENTS

    
    if(kernelDim==3){
        print_timestamp((char*) "START CPU FILTER OPT FLOAT 3x3 PARALLEL");    
    }
    else if(kernelDim==5){
        print_timestamp((char*) "START CPU FILTER OPT FLOAT 5x5 PARALLEL");     
    }
    else{
        printf("ERROR: kernelDim=%d\n", kernelDim);
        return;
    }
    

    for(int t=1; t<=maxThreads; t++){
        // omp_set_num_threads(t);
        
        gettimeofday(&tvCpu, NULL);  
        tmCpu = localtime(&tvCpu.tv_sec);
        startFilter.hour = tmCpu->tm_hour;
        startFilter.minutes = tmCpu->tm_min;
        startFilter.seconds = tmCpu->tm_sec;
        startFilter.msec = (int) (tvCpu.tv_usec / 1000);

        if(kernelDim==3){
            parallelOptFilterCpuFloat_3x3(input, cpu_filtered_frame_parallel_float, frameWidth, frameHeight, kernelIdx, t);
        }
        else{
            parallelOptFilterCpuFloat_5x5(input, cpu_filtered_frame_parallel_float, frameWidth, frameHeight, kernelIdx, t);
        }
        

        gettimeofday(&tvCpu, NULL);
        tmCpu = localtime(&tvCpu.tv_sec);
        finishFilter.hour = tmCpu->tm_hour;
        finishFilter.minutes = tmCpu->tm_min;
        finishFilter.seconds = tmCpu->tm_sec;
        finishFilter.msec = (int) (tvCpu.tv_usec / 1000);
        timeDifference = computeTimeDifferenceMs(startFilter, finishFilter);

        printf(">> PARALLEL [%d threads] CPU FILTER [FLOAT] %dx%d took %d ms\n", t, kernelDim, kernelDim, timeDifference);        
        
    }

    if(kernelDim==3){
        print_timestamp((char*) "END CPU FILTER OPT FLOAT 3x3 PARALLEL");          
    }
    else{
        print_timestamp((char*) "END CPU FILTER OPT FLOAT 5x5 PARALLEL");          
    }






    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    // Report
    if(reportFilterResults){
        // printf("CPU FILTERED SAMPLES SERIAL\n");
        // for(int i=0; i<frameHeight; i++){
        //     for(int j=0; j<frameWidth; j++){
        //         printf("%d,",cpu_filtered_frame[i*frameWidth+j]);
        //     }
        //     printf("\n");
        // }

        printf("CPU FILTERED SAMPLES INT PARALLEL\n");
        for(int i=0; i<frameHeight; i++){
            for(int j=0; j<frameWidth; j++){
                printf("%d,",cpu_filtered_frame_parallel_int[i*frameWidth+j]);
            }
            printf("\n");
        }
        
        printf("\n\n");

        printf("CPU FILTERED SAMPLES FLOAT PARALLEL\n");
        for(int i=0; i<frameHeight; i++){
            for(int j=0; j<frameWidth; j++){
                printf("%d,",cpu_filtered_frame_parallel_float[i*frameWidth+j]);
            }
            printf("\n");
        }
    }
    
    free(cpu_filtered_frame_parallel_int);
    free(cpu_filtered_frame_parallel_float);

}