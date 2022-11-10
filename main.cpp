#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <string.h>
#include <limits.h>
#include <assert.h>
#include <time.h>
#include "main_aux_functions.h"

using namespace std;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)

int main(int argc, char *argv[]) {
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("intra.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    // Error objects for detecting problems in OpenCL
    cl_int error, error_1, error_2, error_3;

    // Get platform and device information
    cl_platform_id *platform_id = NULL;
    cl_uint ret_num_platforms;

    error = clGetPlatformIDs(0, NULL, &ret_num_platforms);
    probe_error(error, (char*)"Error querying available platforms\n");
    
    platform_id = (cl_platform_id*) malloc(sizeof(cl_platform_id) * ret_num_platforms); // Malloc space for all ret_num_platforms platforms
    
    error = clGetPlatformIDs(ret_num_platforms, platform_id, NULL);
    probe_error(error, (char*)"Error querying platform IDs\n");

    // List available platforms
    char platform_name[128] = {0};
    // List name of platforms available and assign to proper CPU/GPU IDs
    cout << "Idx    Platform Name" << endl;
    for (cl_uint ui=0; ui< ret_num_platforms; ++ui){
        error = clGetPlatformInfo(platform_id[ui], CL_PLATFORM_NAME, 128 * sizeof(char), platform_name, NULL);
        probe_error(error, (char*)"Error querying CL_PLATFORM_NAME\n");
        if (platform_name != NULL){
            cout << ui << "      " << platform_name << endl;
        }
    }

    // Scan all platforms looking for CPU and GPU devices.
    // This results in errors when searching for GPU devices on CPU platforms, for instance. Not a problem
    cl_device_id cpu_device_ids[5] = {NULL, NULL, NULL, NULL, NULL};
    cl_uint ret_cpu_num_devices;
    int assigned_cpus = 0; // Keeps number of available devices
    
    cl_device_id gpu_device_ids[5] = {NULL, NULL, NULL, NULL, NULL};
    cl_uint ret_gpu_num_devices;
    int assigned_gpus = 0; // Keeps number of available devices

    cl_device_id tmp_device_ids[5] = {NULL, NULL, NULL, NULL, NULL};

    // Scan all platforms...
    printf("\n");
    for(cl_uint p=0; p<ret_num_platforms; p++){
        error = clGetPlatformInfo(platform_id[p], CL_PLATFORM_NAME, 128 * sizeof(char), platform_name, NULL);
        probe_error(error, (char*)"Error querying CL_PLATFORM_NAME\n");
        printf("Scanning platform %d...\n", p);
     
        // Query all CPU devices on current platform, and copy them to global CPU devices list (cpu_device_ids)
        error = clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_CPU, 0, 
            NULL, &ret_cpu_num_devices);
        error |= clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_CPU, ret_cpu_num_devices, tmp_device_ids, NULL);
        probe_error(error, (char*)"\tError querying CPU device IDs\n"); // GPU platforms do not have CPU devices
        
        for(cl_uint d=0; d<ret_cpu_num_devices; d++){
                cpu_device_ids[assigned_cpus] = tmp_device_ids[d];
                assigned_cpus++;
        }
        
        // Query all GPU devices on current platform, and copy them to global GPU devices list (gpu_device_ids)
        error = clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_GPU, 0, 
            NULL, &ret_gpu_num_devices);
        error |= clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_GPU, ret_gpu_num_devices, tmp_device_ids, NULL);
        probe_error(error, (char*)"\tError querying GPU device IDs\n");  // CPU platforms do not have GPU devices
        
        for(cl_uint d=0; d<ret_gpu_num_devices; d++){
                gpu_device_ids[assigned_gpus] = tmp_device_ids[d];
                assigned_gpus++;
        }
    }
    printf("\n");

    char device_name[1024];
    char device_extensions[1024];

    // List the ID and name for each CPU and GPU device
    for(int cpu=0; cpu<assigned_cpus; cpu++){
        error = clGetDeviceInfo(cpu_device_ids[cpu], CL_DEVICE_NAME, 1024 * sizeof(char), device_name, NULL);
        error&= clGetDeviceInfo(cpu_device_ids[cpu], CL_DEVICE_EXTENSIONS, 1024 * sizeof(char), device_extensions, NULL);
        probe_error(error, (char*)"Error querying CL_DEVICE_NAME\n");
        
        cout << "CPU " << cpu << endl;
        cout << "\tid " << cpu_device_ids[cpu] << endl << "\t" <<  device_name << endl;
        cout << "\tExtensions: " << device_extensions << endl;
    }
    for(int gpu=0; gpu<assigned_gpus; gpu++){
        error = clGetDeviceInfo(gpu_device_ids[gpu], CL_DEVICE_NAME, 1024 * sizeof(char), device_name, NULL);
        probe_error(error, (char*)"Error querying CL_DEVICE_NAME\n");
        error&= clGetDeviceInfo(gpu_device_ids[gpu], CL_DEVICE_EXTENSIONS, 1024 * sizeof(char), device_extensions, NULL);
        
        cout << "GPU " << gpu << endl;
        cout << "\tid " << gpu_device_ids[gpu] << endl << "\t" <<  device_name << endl;
        cout << "\tExtensions: " << device_extensions << endl;
    }

    
    // Create "target" device and assign proper IDs
    cl_device_id device_id = NULL; 
    
    // Select what CPU or GPU will be used based on parameters
    printf("NUMERO DE ARGUMENTOS %d\n", argc);
    if(argc==6){
        if(!strcmp(argv[1],"CPU")){
            if(stoi(argv[2]) < assigned_cpus){
                cout << "COMPUTING ON CPU " << argv[2] << endl;        
                device_id = cpu_device_ids[stoi(argv[2])];    
            }
            else{
                cout << "Incorrect CPU number. Only " << assigned_cpus << " CPUs are detected" << endl;
                exit(0);    
            }
        }
        else if(!strcmp(argv[1],"GPU")){
            if(stoi(argv[2]) < assigned_gpus){
                cout << "COMPUTING ON GPU " << argv[2] << endl;        
                device_id = gpu_device_ids[stoi(argv[2])];    
            }
            else{
                cout << "Incorrect GPU number. Only " << assigned_gpus << " GPUs are detected" << endl;
                exit(0);    
            }
        }
        else{
            cout << "Incorrect usage. First parameter must be either CPU or GPU" << endl;
            exit(0);
        }
    }
    else{
        cout << "\n\n\nFailed to specify the input parameters. Proper execution has the form of" << endl;
        cout << "./main <CPU or GPU> <# of CPU or GPU device> <file with curr block> <file with references> <output file preffix>\n\n\n" << endl;
        exit(0);
    }
    
    size_t ret_val;
    cl_uint max_compute_units;
    error = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &ret_val);
    error|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, ret_val, &max_compute_units, NULL);
    probe_error(error, (char*)"Error querying maximum number of compute units of device\n");
    cout << "-- Max compute units " << max_compute_units << endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////         STARTS BY CREATING A CONTEXT, QUEUE, AND MOVING DATA INTO THE BUFFERS         /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &error);
    probe_error(error, (char*)"Error creating context\n");

    // Create a command queue
    // Profiling enabled to measure execution time. 
    // TODO: Remove this profiling when perform actual computation, it may slowdown the processing (for experiments and etc)
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    // TODO: This should be an input parameter
    const int frameWidth  = 1920; // 1920 or 3840
    const int frameHeight = 1080; // 1080 or 2160

    // TODO: This should be computed based on the frame resolution
    const int nCtus = frameHeight==1080 ? 135 : 510; //135 or 510 for 1080p and 2160p  ||  1080p videos have 120 entire CTUs plus 15 partial CTUs || 4k videos have 480 entire CTUs plus 30 partial CTUs
    const int itemsPerWG = 64;  // Each workgroup has 256 workitems
    int nWG; // All CU sizes inside all CTUs are being processed simultaneously by distinct WGs

    // Read the input data
    string currBlockFileName = argv[3];         // File with samples from current block
    string currReferencesFileName = argv[4];    // File with reference samples for current block
    string outputFilePreffix = argv[5];         // Preffix of exported files containing the prediction signal

    ifstream currBlockFile, currReferenceFile;
    currBlockFile.open(currBlockFileName);
    currReferenceFile.open(currReferencesFileName);

    if (!currBlockFile.is_open() || !currReferenceFile.is_open()) {     // validate file open for reading 
        perror (("error while opening samples files" ));
        return 1;
    }

    string refLine, refVal, currLine, currVal;

    // const int FRAME_SIZE = frameWidth*frameHeight;
    const int BLOCK_WIDTH = 64;
    const int BLOCK_HEIGHT = 64;
    const int BLOCK_SIZE = BLOCK_WIDTH*BLOCK_HEIGHT;
    const int NUM_PRED_MODES = 6;
    const int TEST_TRANSPOSED_MODES = 1;
    const int TOTAL_PREDICTION_MODES = NUM_PRED_MODES * (TEST_TRANSPOSED_MODES ? 2 : 1);
    const int REF_SIZE = 2*BLOCK_WIDTH + 2*BLOCK_HEIGHT + 1;

    unsigned short *currentBlock = (unsigned short*) malloc(sizeof(short) * BLOCK_SIZE);
    unsigned short *references   = (unsigned short*) malloc(sizeof(short) * REF_SIZE);

    // Read the samples from current block into a matrix
    for(int h=0; h<BLOCK_HEIGHT; h++){
        getline(currBlockFile, currLine, '\n');
        stringstream currStream(currLine); 
        for(int w=0; w<BLOCK_WIDTH; w++){
            getline(currStream, currVal, ',');
            currentBlock[h*BLOCK_WIDTH + w] = stoi(currVal);
        }
    }
    // Read th references into an array
    getline(currReferenceFile, refLine, '\n');
    stringstream refStream(refLine); 
    for(int i=0; i<REF_SIZE; i++){
        getline(refStream, refVal, ',');
        references[i] = stoi(refVal);
    }

    
    // These buffers are for storing the reference samples and current samples
    cl_mem currBlock_memObj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            BLOCK_SIZE * sizeof(short), NULL, &error_1);    
    cl_mem refSamples_memObj = clCreateBuffer(context, CL_MEM_READ_ONLY,
            REF_SIZE * sizeof(short), NULL, &error_2);                
    error = error_1 || error_2;

    probe_error(error, (char*)"Error creating memory buffers\n");
    
    double nanoSeconds = 0;
    // These variabels are used to profile the time spend writing to memory objects "clEnqueueWriteBuffer"
    cl_ulong write_time_start;
    cl_ulong write_time_end;
    cl_event write_event;

    error  = clEnqueueWriteBuffer(command_queue, refSamples_memObj, CL_TRUE, 0, 
            REF_SIZE * sizeof(short), references, 0, NULL, &write_event); 
    error = clWaitForEvents(1, &write_event);
    probe_error(error, (char*)"Error waiting for write events\n");  
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing write\n");
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(write_time_start), &write_time_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(write_time_end), &write_time_end, NULL);
    nanoSeconds += write_time_end-write_time_start;

    error |= clEnqueueWriteBuffer(command_queue, currBlock_memObj, CL_TRUE, 0, 
            BLOCK_SIZE * sizeof(short), currentBlock, 0, NULL, &write_event);      
    error = clWaitForEvents(1, &write_event);
    probe_error(error, (char*)"Error waiting for write events\n");  
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing write\n");
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(write_time_start), &write_time_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(write_time_end), &write_time_end, NULL);
    nanoSeconds += write_time_end-write_time_start;

    writeTime = nanoSeconds;
    nanoSeconds = 0.0;

    probe_error(error, (char*)"Error copying data from memory to buffers LEGACY\n");

    
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////       CREATE A PROGRAM (OBJECT) BASED ON .cl FILE AND BUILD IT TO TARGET DEVICE       /////
    /////         CREATE A KERNEL BY ASSIGNING A NAME FOR THE RECENTLY COMPILED PROGRAM         /////
    /////                           LOADS THE ARGUMENTS FOR THE KERNEL                          /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Create a program from the kernel source (specifically for the device in the context variable)
    cl_program program = clCreateProgramWithSource(context, 1, 
            (const char **)&source_str, (const size_t *)&source_size, &error);
    probe_error(error, (char*)"Error creating program from source\n");

    // Build the program
    error = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    probe_error(error, (char*)"Error building the program\n");
    // Show debugging information when the build is not successful
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *) malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////                  VARIABLES SHARED BETWEEN THE EXECUTION OF ALL KERNELS                /////
    /////           THIS INCLUDES CONSTANTS, VARIABLES USED FOR CONTROL AND PROFILING           /////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Used to access optimal workgroup size
    size_t size_ret;
    cl_uint preferred_size, maximum_size;

    cl_kernel kernel; // The object that holds the compiled kernel to be enqueued

    // Used to profile execution time of kernel
    cl_event event;
    cl_ulong time_start, time_end;

    // Used to set the workgroup sizes
    size_t global_item_size, local_item_size;

    // Used to export the kernel results into proper files or the terminal
    string exportFileName;
    
    int reportToTerminal = 1;
    int reportToFile = 0;


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////              THESE MEMORY OBJECTS AND ARRAYS MUST BE FREED AFTER EXECUTION            /////
    /////////////////////////////////////////////////////////////////////////////////////////////////    
    // These memory objects hold the predicted signal and distortion after the kernel has finished
    cl_mem return_predictedBlock_memObj;
    cl_mem return_SATD_memObj;
    cl_mem return_SAD_memObj;

    // These memory objects are used to store intermediate data and debugging information from the kernel
    cl_mem debug_mem_obj;
      
    // These dynamic arrays retrieve the information from the kernel to the host
    // Predicted signal and distortion
    short *return_predictedBlock;
    long *return_SATD, *return_SAD;

    // Debug information returned by kernel
    short *debug_data;


    // TODO: Correct to the proper number of WGs and CTUs
    // Currently, each WG will apply one prediction mode TO ONE cu, and each workitem will predict a set of samples
    nWG = TOTAL_PREDICTION_MODES; 
    
    // -----------------------------
    // Allocate some memory space
    // -----------------------------
    return_predictedBlock = (short*) malloc(sizeof(short) * BLOCK_SIZE * TOTAL_PREDICTION_MODES);
    return_SATD = (long*) malloc(sizeof(long) * TOTAL_PREDICTION_MODES);
    return_SAD = (long*) malloc(sizeof(long) * TOTAL_PREDICTION_MODES);
    // Debug information returned by kernel
    debug_data = (short*)  malloc(sizeof(short)  * nWG*itemsPerWG*4);

    return_predictedBlock_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            BLOCK_SIZE * TOTAL_PREDICTION_MODES * sizeof(short), NULL, &error_1);
    return_SATD_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            BLOCK_SIZE * TOTAL_PREDICTION_MODES * sizeof(long), NULL, &error_2);
    return_SAD_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            BLOCK_SIZE * TOTAL_PREDICTION_MODES * sizeof(long), NULL, &error_3);                        
    error = error_1 | error_2 | error_3;
    probe_error(error,(char*)"Error creating memory object for predicted CU and costs\n");

    // These memory objects are used to store intermediate data and debugging information from the kernel
    debug_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
            nWG*itemsPerWG*4 * sizeof(cl_short), NULL, &error_1);  

    error = error_1;
    probe_error(error,(char*)"Error creating memory object for debugging information\n");


    // Create kernel
    kernel = clCreateKernel(program, "MIP_64x64", &error);
    probe_error(error, (char*)"Error creating MIP_64x64 kernel\n"); 
    printf("Performing MIP_64x64 kernel...\n");

    // Query for work groups sizes information
    error = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
    error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);
    
    probe_error(error, (char*)"Error querying preferred or maximum work group size\n");
    cout << "-- Preferred WG size multiple " << preferred_size << endl;
    cout << "-- Maximum WG size " << maximum_size << endl;

    // Set the arguments of the kernel
    error_1  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&currBlock_memObj);
    error_1 |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&refSamples_memObj);
    error_1 |= clSetKernelArg(kernel, 2, sizeof(cl_int), (void *)&TOTAL_PREDICTION_MODES);
    error_1 |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&return_predictedBlock_memObj);
    error_1 |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&return_SATD_memObj);
    error_1 |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&return_SAD_memObj);   
    error_1 |= clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&debug_mem_obj);   
    probe_error(error_1, (char*)"Error setting arguments for the kernel\n");

    // Execute the OpenCL kernel on the list
    // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
    global_item_size = nWG*itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
    local_item_size = itemsPerWG; 
    
    error = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
            &global_item_size, &local_item_size, 0, NULL, &event);
    probe_error(error, (char*)"Error enqueuing kernel\n");

    error = clWaitForEvents(1, &event);
    probe_error(error, (char*)"Error waiting for events\n");
    
    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing\n");

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end-time_start;
    
    execTime = nanoSeconds;
    nanoSeconds = 0;

    // Read affine results from memory objects into host arrays
    readMemobjsIntoArray(command_queue, TOTAL_PREDICTION_MODES, BLOCK_SIZE, return_predictedBlock_memObj, return_SATD_memObj, return_SAD_memObj, return_predictedBlock, return_SATD, return_SAD, debug_mem_obj, debug_data);
    reportTimingResults();
    
    for(int mode=0; mode<TOTAL_PREDICTION_MODES; mode++){
        printf("Final prediction (upsampled) of mode %d (MODE %d Transp %d)\n\n", mode, mode%6, mode/6);
        for(int i=0; i<64; i++){
            for(int j=0; j<64; j++){
                printf("%d,", return_predictedBlock[mode*BLOCK_SIZE + i*64+j]);
            }
            printf("\n");
        }
        printf("\n-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n");
    }


    // printf("Reduced boundaries for mode %d\n", mode);
    // for(int i=0; i<8; i++){
    //     printf("%d,", debug_data[i]);
    // }
    // printf("\n");


    // -----------------------------------------------------------------
    //
    //  MANAGE ANY DEBUGGING INFORMATION FROM THE KERNEL
    //
    // -----------------------------------------------------------------

    /* Print the contents of debug_data. BEWARE of the data types (long, short, int, ...)
    printf("Debug array...\n");
    // for(int i=0; i<nWG*itemsPerWG; i++){
    for(int j=0; j<128; j++){
        for(int k=0; k<128; k++){
            int iC[4];
            iC[0] = debug_data[j*128*4 + k*4 + 0];
            iC[1] = debug_data[j*128*4 + k*4 + 1];
            iC[2] = debug_data[j*128*4 + k*4 + 2];
            iC[3] = debug_data[j*128*4 + k*4 + 3];
            printf("j=%d,k=%d,iC[0]=%d,iC[1]=%d,iC[2]=%d,iC[3]=%d\n", j, k, iC[0], iC[1], iC[2], iC[3]);
        }
    //    printf("[%d] = %ld -> %d\n", i, debug_data[2*i], debug_data[2*i+1]);
    }
    //*/

    ////////////////////////////////////////////////////
    /////         FREE SOME MEMORY SPACE           /////
    ////////////////////////////////////////////////////

    // Clean up
    error = clFlush(command_queue);
    error |= clFinish(command_queue);
    error |= clReleaseKernel(kernel);
    error |= clReleaseProgram(program);
    error |= clReleaseCommandQueue(command_queue);
    error |= clReleaseContext(context);
    error |= clReleaseMemObject(currBlock_memObj);
    error |= clReleaseMemObject(refSamples_memObj);
    error |= clReleaseMemObject(return_predictedBlock_memObj);
    error |= clReleaseMemObject(return_SATD_memObj);
    error |= clReleaseMemObject(return_SAD_memObj);
    error |= clReleaseMemObject(debug_mem_obj);
    probe_error(error, (char*)"Error releasing  OpenCL objects\n");
    
    free(source_str);
    free(platform_id);
    free(currentBlock);
    free(references);
    free(return_predictedBlock);
    free(return_SATD);
    free(return_SAD);
    free(debug_data);

    return 0;
}