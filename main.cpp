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

int main(int argc, char *argv[])
{
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    fp = fopen("intra.cl", "r");
    if (!fp)
    {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
 
    // Error objects for detecting problems in OpenCL
    cl_int error, error_1, error_2, error_3, error_4;

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
    for (cl_uint ui = 0; ui < ret_num_platforms; ++ui)
    {
        error = clGetPlatformInfo(platform_id[ui], CL_PLATFORM_NAME, 128 * sizeof(char), platform_name, NULL);
        probe_error(error, (char*)"Error querying CL_PLATFORM_NAME\n");
        if (platform_name != NULL)
        {
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
    for (cl_uint p = 0; p < ret_num_platforms; p++)
    {
        error = clGetPlatformInfo(platform_id[p], CL_PLATFORM_NAME, 128 * sizeof(char), platform_name, NULL);
        probe_error(error, (char*)"Error querying CL_PLATFORM_NAME\n");
        printf("Scanning platform %d...\n", p);

        // Query all CPU devices on current platform, and copy them to global CPU devices list (cpu_device_ids)
        error = clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_CPU, 0, 
            NULL, &ret_cpu_num_devices);
        error |= clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_CPU, ret_cpu_num_devices, tmp_device_ids, NULL);
        probe_error(error, (char*)"\tError querying CPU device IDs\n"); // GPU platforms do not have CPU devices

        for (cl_uint d = 0; d < ret_cpu_num_devices; d++)
        {
            cpu_device_ids[assigned_cpus] = tmp_device_ids[d];
            assigned_cpus++;
        }

        // Query all GPU devices on current platform, and copy them to global GPU devices list (gpu_device_ids)
        error = clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_GPU, 0, 
            NULL, &ret_gpu_num_devices);
        error |= clGetDeviceIDs( platform_id[p], CL_DEVICE_TYPE_GPU, ret_gpu_num_devices, tmp_device_ids, NULL);
        probe_error(error, (char*)"\tError querying GPU device IDs\n");  // CPU platforms do not have GPU devices

        for (cl_uint d = 0; d < ret_gpu_num_devices; d++)
        {
                gpu_device_ids[assigned_gpus] = tmp_device_ids[d];
                assigned_gpus++;
        }
    }
    printf("\n");

    char device_name[1024];
    char device_extensions[1024];

    // List the ID and name for each CPU and GPU device
    for (int cpu = 0; cpu < assigned_cpus; cpu++)
    {
        error = clGetDeviceInfo(cpu_device_ids[cpu], CL_DEVICE_NAME, 1024 * sizeof(char), device_name, NULL);
        error&= clGetDeviceInfo(cpu_device_ids[cpu], CL_DEVICE_EXTENSIONS, 1024 * sizeof(char), device_extensions, NULL);
        probe_error(error, (char*)"Error querying CL_DEVICE_NAME\n");

        cout << "CPU " << cpu << endl;
        cout << "\tid " << cpu_device_ids[cpu] << endl << "\t" <<  device_name << endl;
        cout << "\tExtensions: " << device_extensions << endl;
    }
    for (int gpu = 0; gpu < assigned_gpus; gpu++)
    {
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
    if(argc==5){
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
        cout << "./main <CPU or GPU> <# of CPU or GPU device> <file with frames of reference frame> <output file preffix>\n\n\n" << endl;
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

    const int itemsPerWG_obtainReducedBoundaries = 128;
    const int itemsPerWG_obtainReducedPrediction = 256;
    const int itemsPerWG_upsampleDistortion = 256;

    int itemsPerWG;
    int nWG; // All CU sizes inside all CTUs are being processed simultaneously by distinct WGs

    // Read the input data
    string refFrameFileName = argv[3];

    string outputFilePreffix = argv[4]; // Preffix of exported files containing the prediction signal

    ifstream refFile;
    refFile.open(refFrameFileName);

    if (!refFile.is_open())
    { // validate file open for reading
        perror(("error while opening samples files"));
        return 1;
    }

    string refLine, refVal, currLine, currVal;

    const int FRAME_SIZE = frameWidth*frameHeight;
    const int nCTUs = 135; //frameHeight==1080 ? 135 : 510; //135 or 510 for 1080p and 2160p  ||  1080p videos have 120 entire CTUs plus 15 partial CTUs || 4k videos have 480 entire CTUs plus 30 partial CTUs
    const int NUM_PRED_MODES = PREDICTION_MODES_ID2;
    const int TEST_TRANSPOSED_MODES = 1;
    const int TOTAL_PREDICTION_MODES = NUM_PRED_MODES * (TEST_TRANSPOSED_MODES ? 2 : 1);

    unsigned short *reference_frame = (unsigned short *)malloc(sizeof(short) * FRAME_SIZE);

    // Read the samples from reference frame into the reference array
    for (int h=0; h<frameHeight; h++)
    {
        getline(refFile, refLine, '\n');
        stringstream currStream(currLine), refStream(refLine); 

        for (int w = 0; w < frameWidth; w++)
        {
            getline(currStream, currVal, ',');
            getline(refStream, refVal, ',');
            reference_frame[h*frameWidth + w] = stoi(refVal);
        }
    }

    // These buffers are used to store the reduced boundaries for all CU sizes
    error = 0;


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////              THESE MEMORY OBJECTS AND ARRAYS MUST BE FREED AFTER EXECUTION            /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // These memory objects are used to store intermediate data and debugging information from the kernel
    cl_mem debug_mem_obj;

    //  *4 is the 4 references in each reduced boundary for SizeID=2
    cl_mem redT_64x64_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * cusPerCtu[_64x64] * 4 * sizeof(short), NULL, &error_1);
    cl_mem redL_64x64_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * cusPerCtu[_64x64] * 4 * sizeof(short), NULL, &error_2);
    error = error || error_1 || error_2;

    cl_mem redT_32x32_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * cusPerCtu[_32x32] * 4 * sizeof(short), NULL, &error_1);
    cl_mem redL_32x32_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * cusPerCtu[_32x32] * 4 * sizeof(short), NULL, &error_2);
    error = error || error_1 || error_2;

    cl_mem redT_16x16_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * cusPerCtu[_16x16] * 4 * sizeof(short), NULL, &error_1);
    cl_mem redL_16x16_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * cusPerCtu[_16x16] * 4 * sizeof(short), NULL, &error_2);
    error = error || error_1 || error_2;

    cl_mem refT_64x64_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * 2 * 128 * sizeof(short), NULL, &error_1);
    cl_mem refL_64x64_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * 2 * 128 * sizeof(short), NULL, &error_2);
    error = error || error_1 || error_2;
    
    cl_mem refT_32x32_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * 4 * 128 * sizeof(short), NULL, &error_1);
    cl_mem refL_32x32_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * 4 * 128 * sizeof(short), NULL, &error_2);
    error = error || error_1 || error_2;

    cl_mem refT_16x16_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * 8 * 128 * sizeof(short), NULL, &error_1);
    cl_mem refL_16x16_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              nCTUs * 8 * 128 * sizeof(short), NULL, &error_2);
    error = error || error_1 || error_2;


    cl_mem referenceFrame_memObj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                  FRAME_SIZE * sizeof(short), NULL, &error_1);

    // These memory objects hold the predicted signal and distortion after the kernel has finished
    cl_mem return_predictionSignal_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                           nCTUs * TOTAL_CUS_PER_CTU * REDUCED_PRED_SIZE_Id2 * REDUCED_PRED_SIZE_Id2 * PREDICTION_MODES_ID2 * 2 * sizeof(short), NULL, &error_2); // Each CTU is composed of TOTAL_CUS_PER_CTU CUs, and each reduced CU is 8*8, and we have 12 prediction modes

    cl_mem return_SATD_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                               nCTUs * TOTAL_CUS_PER_CTU * PREDICTION_MODES_ID2 * 2 * sizeof(long), NULL, &error_3);
    cl_mem return_SAD_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                               nCTUs * TOTAL_CUS_PER_CTU * PREDICTION_MODES_ID2 * 2 * sizeof(long), NULL, &error_4);

    error = error || error_1 || error_2 || error_3 || error_4;

    probe_error(error, (char*)"Error creating memory buffers\n");

    double nanoSeconds = 0;
    // These variabels are used to profile the time spend writing to memory objects "clEnqueueWriteBuffer"
    cl_ulong write_time_start;
    cl_ulong write_time_end;
    cl_event write_event;

    error = clEnqueueWriteBuffer(command_queue, referenceFrame_memObj, CL_TRUE, 0,
                                 FRAME_SIZE * sizeof(short), reference_frame, 0, NULL, &write_event);
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
    if (error == CL_BUILD_PROGRAM_FAILURE)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *)malloc(log_size);

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

    cl_kernel kernel_initRefSamples, kernel_reducedPrediction, kernel_upsampleDistortion; // The object that holds the compiled kernel to be enqueued

    // Used to profile execution time of kernel
    cl_event event;
    cl_ulong time_start, time_end;

    // Used to set the workgroup sizes
    size_t global_item_size, local_item_size;

    // Used to export the kernel results into proper files or the terminal
    string exportFileName;

    int reportToTerminal = 1;
    int reportDistortionOnlyTarget = 0;
    int reportToFile = 0;
    int targetCTU = 16;

    ///////////////////////////////////////////////////////////////////////////////////////
    /////              THESE DYNAMIC ARRAYS  MUST BE FREED AFTER EXECUTION            /////
    ///////////////////////////////////////////////////////////////////////////////////////

    // These dynamic arrays retrieve the information from the kernel to the host
    // Predicted signal and distortion
    short *return_reducedPredictionSignal;
    long *return_SATD, *return_SAD;
    short *return_redT_64x64, *return_redL_64x64, *return_redT_32x32, *return_redL_32x32, *return_redT_16x16, *return_redL_16x16;
    short *return_refT_64x64, *return_refL_64x64, *return_refT_32x32, *return_refL_32x32, *return_refT_16x16, *return_refL_16x16;

    // Debug information returned by kernel
    short *debug_data;

    // TODO: Correct to the proper number of WGs and CTUs
    // Currently, each WG will process one CTU partitioned into one CU size, and produce the reduced boundaries for each one of the CUs
    nWG = nCTUs*NUM_CU_SIZES;

    debug_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   nWG * itemsPerWG_upsampleDistortion * 4 * sizeof(cl_short), NULL, &error_1);

    // -----------------------------
    // Allocate some memory space
    // -----------------------------
    return_reducedPredictionSignal = (short*)malloc(sizeof(short) * nCTUs * 8 * 8 * TOTAL_CUS_PER_CTU * PREDICTION_MODES_ID2*2); // Each predicted CU has 8x8 samples
    return_SATD = (long*) malloc(sizeof(long) * nCTUs * TOTAL_CUS_PER_CTU * PREDICTION_MODES_ID2*2);
    return_SAD = (long*) malloc(sizeof(long) * nCTUs * TOTAL_CUS_PER_CTU * PREDICTION_MODES_ID2*2);
    // Reduced boundaries
    return_redT_64x64 = (short*) malloc(sizeof(short) * nCTUs * cusPerCtu[_64x64] * 4);
    return_redL_64x64 = (short*) malloc(sizeof(short) * nCTUs * cusPerCtu[_64x64] * 4);
    return_redT_32x32 = (short*) malloc(sizeof(short) * nCTUs * cusPerCtu[_32x32] * 4);
    return_redL_32x32 = (short*) malloc(sizeof(short) * nCTUs * cusPerCtu[_32x32] * 4);
    return_redT_16x16 = (short*) malloc(sizeof(short) * nCTUs * cusPerCtu[_16x16] * 4);
    return_redL_16x16 = (short*) malloc(sizeof(short) * nCTUs * cusPerCtu[_16x16] * 4);
    // Complete boundaries
    return_refT_64x64 = (short*) malloc(sizeof(short) * nCTUs * cuRowsPerCtu[_64x64] * 128);
    return_refL_64x64 = (short*) malloc(sizeof(short) * nCTUs * cuColumnsPerCtu[_64x64] * 128);
    return_refT_32x32 = (short*) malloc(sizeof(short) * nCTUs * cuRowsPerCtu[_32x32] * 128);
    return_refL_32x32 = (short*) malloc(sizeof(short) * nCTUs * cuColumnsPerCtu[_32x32] * 128);
    return_refT_16x16 = (short*) malloc(sizeof(short) * nCTUs * cuRowsPerCtu[_16x16] * 128);
    return_refL_16x16 = (short*) malloc(sizeof(short) * nCTUs * cuColumnsPerCtu[_16x16] * 128);
    // Debug information returned by kernel
    debug_data = (short*) malloc(sizeof(short) * nWG*itemsPerWG_upsampleDistortion*4);

    // These memory objects are used to store intermediate data and debugging information from the kernel
    debug_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   nWG * itemsPerWG_upsampleDistortion * 4 * sizeof(cl_short), NULL, &error_1);

    error = error_1;
    probe_error(error,(char*)"Error creating memory object for debugging information\n");

    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //          FIRST WE MUST OBTAIN THE REDUCED BOUNDARIES
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    itemsPerWG = itemsPerWG_obtainReducedBoundaries;

    // Create kernel
    kernel_initRefSamples = clCreateKernel(program, "initBoundariesSquareSizeId2", &error);
    probe_error(error, (char*)"Error creating initBoundariesSquareSizeId2 kernel\n"); 
    printf("Performing initBoundariesSquareSizeId2 kernel...\n");

    // Query for work groups sizes information
    error = clGetKernelWorkGroupInfo(kernel_initRefSamples, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel_initRefSamples, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
    error |= clGetKernelWorkGroupInfo(kernel_initRefSamples, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel_initRefSamples, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

    probe_error(error, (char*)"Error querying preferred or maximum work group size\n");
    cout << "-- Preferred WG size multiple " << preferred_size << endl;
    cout << "-- Maximum WG size " << maximum_size << endl;

    // Set the arguments of the kernel initBoundariesSquareSizeId2
    error_1 = clSetKernelArg(kernel_initRefSamples, 0, sizeof(cl_mem), (void *)&referenceFrame_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 1, sizeof(cl_int), (void *)&frameWidth);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 2, sizeof(cl_int), (void *)&frameHeight);
    // Reduced boundaries
    error_1 |= clSetKernelArg(kernel_initRefSamples, 3, sizeof(cl_mem), (void *)&redT_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 4, sizeof(cl_mem), (void *)&redL_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 5, sizeof(cl_mem), (void *)&redT_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 6, sizeof(cl_mem), (void *)&redL_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 7, sizeof(cl_mem), (void *)&redT_16x16_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 8, sizeof(cl_mem), (void *)&redL_16x16_memObj);
    // Complete boundaries
    error_1 |= clSetKernelArg(kernel_initRefSamples, 9, sizeof(cl_mem),  (void *)&refT_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 10, sizeof(cl_mem), (void *)&refL_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 11, sizeof(cl_mem), (void *)&refT_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 12, sizeof(cl_mem), (void *)&refL_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 13, sizeof(cl_mem), (void *)&refT_16x16_memObj);
    error_1 |= clSetKernelArg(kernel_initRefSamples, 14, sizeof(cl_mem), (void *)&refL_16x16_memObj);

    probe_error(error_1, (char*)"Error setting arguments for the kernel\n");

    // Execute the OpenCL kernel on the list
    // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
    global_item_size = nWG*itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
    local_item_size = itemsPerWG;

    error = clEnqueueNDRangeKernel(command_queue, kernel_initRefSamples, 1, NULL,
                                   &global_item_size, &local_item_size, 0, NULL, &event);
    probe_error(error, (char*)"Error enqueuing kernel\n");

    error = clWaitForEvents(1, &event);
    probe_error(error, (char*)"Error waiting for events\n");

    error = clFinish(command_queue);
    probe_error(error, (char*)"Error finishing\n");

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end-time_start;

    execTime_reducedBoundaries = nanoSeconds;

    execTime = execTime_reducedBoundaries;
    nanoSeconds = 0;

    // Read affine results from memory objects into host arrays
    readMemobjsIntoArray_ReducedBoundaries(command_queue, nCTUs, redT_64x64_memObj, redL_64x64_memObj, redT_32x32_memObj, redL_32x32_memObj, redT_16x16_memObj, redL_16x16_memObj, return_redT_64x64, return_redL_64x64, return_redT_32x32, return_redL_32x32, return_redT_16x16, return_redL_16x16);
    readMemobjsIntoArray_CompleteBoundaries(command_queue, nCTUs, refT_64x64_memObj, refL_64x64_memObj, refT_32x32_memObj, refL_32x32_memObj, refT_16x16_memObj, refL_16x16_memObj, return_refT_64x64, return_refL_64x64, return_refT_32x32, return_refL_32x32, return_refT_16x16, return_refL_16x16);
    
    // Export the reduced boundaries for all CU sizes inside a target CTU index
    if (0 && reportToTerminal) {
        // Export the reduced boundaries for all CU sizes inside a target CTU index
        printf("=-=-=-=-=- RESULTS FOR CTU %d @(%dx%d)\n", targetCTU, 128 * (targetCTU % 15), 128 * (targetCTU / 15));
        printf("=-=-=-=-=- REDUCED TOP BOUNDARIES RESULTS -=-=-=-=-=\n");
        printf("RESULTS FOR 64x64\n");
        for (int cu = 0; cu < 4; cu++)
        {
            printf("CU %d\n", cu);
            printf("%d,%d,%d,%d,\n", return_redT_64x64[targetCTU * cusPerCtu[_64x64] * 4 + cu * 4 + 0], return_redT_64x64[targetCTU * cusPerCtu[_64x64] * 4 + cu * 4 + 1], return_redT_64x64[targetCTU * cusPerCtu[_64x64] * 4 + cu * 4 + 2], return_redT_64x64[targetCTU * cusPerCtu[_64x64] * 4 + cu * 4 + 3]);
        }
        printf("\n");
        printf("RESULTS FOR 32x32\n");
        for(int cu=0; cu<16; cu++){
            printf("CU %d\n", cu);
            printf("%d,%d,%d,%d,\n", return_redT_32x32[targetCTU*cusPerCtu[_32x32]*4 + cu*4 + 0], return_redT_32x32[targetCTU*cusPerCtu[_32x32]*4 + cu*4 + 1], return_redT_32x32[targetCTU*cusPerCtu[_32x32]*4 + cu*4 + 2], return_redT_32x32[targetCTU*cusPerCtu[_32x32]*4 + cu*4 + 3]);
        }
        printf("\n");
        printf("RESULTS FOR 16x16\n");
        for(int cu=0; cu<64; cu++){
            printf("CU %d\n", cu);
            printf("%d,%d,%d,%d,\n", return_redT_16x16[targetCTU*cusPerCtu[_16x16]*4 + cu*4 + 0], return_redT_16x16[targetCTU*cusPerCtu[_16x16]*4 + cu*4 + 1], return_redT_16x16[targetCTU*cusPerCtu[_16x16]*4 + cu*4 + 2], return_redT_16x16[targetCTU*cusPerCtu[_16x16]*4 + cu*4 + 3]);
        }
        printf("\n\n\n");
        printf("=-=-=-=-=- REDUCED LEFT BOUNDARIES RESULTS -=-=-=-=-=\n");
        printf("RESULTS FOR 64x64\n");
        for (int cu = 0; cu < 4; cu++)
        {
            printf("CU %d\n", cu);
            printf("%d,%d,%d,%d,\n", return_redL_64x64[targetCTU * cusPerCtu[_64x64] * 4 + cu * 4 + 0], return_redL_64x64[targetCTU * cusPerCtu[_64x64] * 4 + cu * 4 + 1], return_redL_64x64[targetCTU * cusPerCtu[_64x64] * 4 + cu * 4 + 2], return_redL_64x64[targetCTU * cusPerCtu[_64x64] * 4 + cu * 4 + 3]);
        }
        printf("\n");
        printf("RESULTS FOR 32x32\n");
        for(int cu=0; cu<16; cu++){
            printf("CU %d\n", cu);
            printf("%d,%d,%d,%d,\n", return_redL_32x32[targetCTU*cusPerCtu[_32x32]*4 + cu*4 + 0], return_redL_32x32[targetCTU*cusPerCtu[_32x32]*4 + cu*4 + 1], return_redL_32x32[targetCTU*cusPerCtu[_32x32]*4 + cu*4 + 2], return_redL_32x32[targetCTU*cusPerCtu[_32x32]*4 + cu*4 + 3]);
        }
        printf("\n");
        printf("RESULTS FOR 16x16\n");
        for(int cu=0; cu<64; cu++){
            printf("CU %d\n", cu);
            printf("%d,%d,%d,%d,\n", return_redL_16x16[targetCTU*cusPerCtu[_16x16]*4 + cu*4 + 0], return_redL_16x16[targetCTU*cusPerCtu[_16x16]*4 + cu*4 + 1], return_redL_16x16[targetCTU*cusPerCtu[_16x16]*4 + cu*4 + 2], return_redL_16x16[targetCTU*cusPerCtu[_16x16]*4 + cu*4 + 3]);
        }
    }
    
    // Export the complete boundaries for all CU sizes inside a target CTU index
    if (0 && reportToTerminal) {
        // Export the complete boundaries for all CU sizes inside a target CTU index
        printf("=-=-=-=-=- RESULTS FOR CTU %d @(%dx%d)\n", targetCTU, 128 * (targetCTU % 15), 128 * (targetCTU / 15));
        printf("=-=-=-=-=- COMPLETE TOP BOUNDARIES RESULTS -=-=-=-=-=\n");
        printf("RESULTS FOR 64x64\n");
        for (int cu = 0; cu < 4; cu++)
        {
            printf("CU %d\n", cu);
            for(int sample=0; sample<64; sample++){
                printf("%d,", return_refT_64x64[targetCTU * cusPerCtu[_64x64] * 64 + cu * 64 + sample]);
            }
            printf("\n");
        }
        printf("\n");
        printf("RESULTS FOR 32x32\n");
        for(int cu=0; cu<16; cu++)
        {
            printf("CU %d\n", cu);
            for(int sample=0; sample<32; sample++){
                printf("%d,", return_refT_32x32[targetCTU * cusPerCtu[_32x32] * 32 + cu * 32 + sample]);
            }
            printf("\n");
        }
        printf("\n");
        printf("RESULTS FOR 16x16\n");
        for(int cu=0; cu<64; cu++)
        {
            printf("CU %d\n", cu);
            for(int sample=0; sample<16; sample++){
                printf("%d,", return_refT_16x16[targetCTU * cusPerCtu[_16x16] * 16 + cu * 16 + sample]);
            }
            printf("\n");
        }
        printf("\n\n\n");
        printf("=-=-=-=-=- COMPLETE LEFT BOUNDARIES RESULTS -=-=-=-=-=\n");
        printf("RESULTS FOR 64x64\n");
        for (int cu = 0; cu < 4; cu++)
        {
            printf("CU %d\n", cu);
            for(int sample=0; sample<64; sample++){
                printf("%d,", return_refL_64x64[targetCTU * cusPerCtu[_64x64] * 64 + cu * 64 + sample]);
            }
            printf("\n");
        }
        printf("\n");
        printf("RESULTS FOR 32x32\n");
        for(int cu=0; cu<16; cu++)
        {
            printf("CU %d\n", cu);
            for(int sample=0; sample<32; sample++){
                printf("%d,", return_refL_32x32[targetCTU * cusPerCtu[_32x32] * 32 + cu * 32 + sample]);
            }
            printf("\n");
        }
        printf("\n");
        printf("RESULTS FOR 16x16\n");
        for(int cu=0; cu<64; cu++)
        {
            printf("CU %d\n", cu);
            for(int sample=0; sample<16; sample++){
                printf("%d,", return_refL_16x16[targetCTU * cusPerCtu[_16x16] * 16 + cu * 16 + sample]);
            }
            printf("\n");
        }
    }
    
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //          NOW WE OBTAIN THE REDUCED PREDICTION FOR ALL CU SIZES AND PREDICTION MODES
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    itemsPerWG = itemsPerWG_obtainReducedPrediction;

    // Create kernel
    kernel_reducedPrediction = clCreateKernel(program, "MIP_squareSizeId2", &error);
    probe_error(error, (char *)"Error creating MIP_squareSizeId2 kernel\n");
    printf("Performing MIP_squareSizeId2 kernel...\n");

    // Query for work groups sizes information
    error = clGetKernelWorkGroupInfo(kernel_reducedPrediction, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel_reducedPrediction, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
    error |= clGetKernelWorkGroupInfo(kernel_reducedPrediction, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel_reducedPrediction, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

    probe_error(error, (char *)"Error querying preferred or maximum work group size\n");
    cout << "-- Preferred WG size multiple " << preferred_size << endl;
    cout << "-- Maximum WG size " << maximum_size << endl;

    // Set the arguments of the MIP_squareSizeId2 kernel
    error_1  = clSetKernelArg(kernel_reducedPrediction, 0, sizeof(cl_mem), (void *)&return_predictionSignal_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 1, sizeof(cl_int), (void *)&frameWidth);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 2, sizeof(cl_int), (void *)&frameHeight);
    // Reduced boundaries
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 3, sizeof(cl_mem), (void *)&redT_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 4, sizeof(cl_mem), (void *)&redL_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 5, sizeof(cl_mem), (void *)&redT_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 6, sizeof(cl_mem), (void *)&redL_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 7, sizeof(cl_mem), (void *)&redT_16x16_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 8, sizeof(cl_mem), (void *)&redL_16x16_memObj);
    // Complete boundaries
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 9, sizeof(cl_mem), (void *)&refT_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 10, sizeof(cl_mem), (void *)&refL_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 11, sizeof(cl_mem), (void *)&refT_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 12, sizeof(cl_mem), (void *)&refL_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 13, sizeof(cl_mem), (void *)&refT_16x16_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 14, sizeof(cl_mem), (void *)&refL_16x16_memObj);

    error_1 |= clSetKernelArg(kernel_reducedPrediction, 15, sizeof(cl_mem), (void *)&return_SAD_memObj);
    error_1 |= clSetKernelArg(kernel_reducedPrediction, 16, sizeof(cl_mem), (void *)&referenceFrame_memObj);

    probe_error(error_1, (char *)"Error setting arguments for the kernel\n");

    // Execute the OpenCL kernel on the list
    // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
    global_item_size = nWG * itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
    local_item_size = itemsPerWG;

    error = clEnqueueNDRangeKernel(command_queue, kernel_reducedPrediction, 1, NULL,
                                   &global_item_size, &local_item_size, 0, NULL, &event);
    probe_error(error, (char *)"Error enqueuing kernel\n");

    error = clWaitForEvents(1, &event);
    probe_error(error, (char *)"Error waiting for events\n");

    error = clFinish(command_queue);
    probe_error(error, (char *)"Error finishing\n");

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end - time_start;

    execTime_reducedPrediction = nanoSeconds;

    readMemobjsIntoArray_reducedPrediction(command_queue, nCTUs, TOTAL_PREDICTION_MODES, return_predictionSignal_memObj,  return_reducedPredictionSignal, return_SAD_memObj, return_SAD);

    if( 0 && reportToTerminal ){
        int cuSizeIdx;

        printf("TRACING REDUCED PREDICTION SIGNAL FOR CTU %d\n", targetCTU);
        printf("      RESULTS FOR CUs 64x64\n");
        cuSizeIdx = _64x64;
        for(int cu=0; cu<cusPerCtu[cuSizeIdx]; cu++){
            for(int m=0; m<12; m++){
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
                        printf("%d,", return_reducedPredictionSignal[currCuModeIdx + i*8 + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }

        printf("      RESULTS FOR CUs 32x32\n");
        cuSizeIdx = _32x32;
        for(int cu=0; cu<cusPerCtu[cuSizeIdx]; cu++){
            for(int m=0; m<12; m++){
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
                        printf("%d,", return_reducedPredictionSignal[currCuModeIdx + i*8 + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        } 

        printf("      RESULTS FOR CUs 16x16\n");
        cuSizeIdx = _16x16;
        for(int cu=0; cu<cusPerCtu[cuSizeIdx]; cu++){
            for(int m=0; m<12; m++){
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
                        printf("%d,", return_reducedPredictionSignal[currCuModeIdx + i*8 + j]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
        }                 
    }


    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    //
    //          FINALLY, UPSAMPLE THE REDUCED PREDICTION AND COMPUTE THE DISTOTION
    //
    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

    itemsPerWG = itemsPerWG_upsampleDistortion;

    // Create kernel
    kernel_upsampleDistortion = clCreateKernel(program, "upsampleDistortionSizeId2", &error);
    probe_error(error, (char *)"Error creating upsampleDistortionSizeId2 kernel\n");
    printf("Performing upsampleDistortionSizeId2 kernel...\n");

    // Query for work groups sizes information
    error = clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
    error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
    error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

    probe_error(error, (char *)"Error querying preferred or maximum work group size\n");
    cout << "-- Preferred WG size multiple " << preferred_size << endl;
    cout << "-- Maximum WG size " << maximum_size << endl;

    // Set the arguments of the upsampleDistortionSizeId2 kernel
    error_1  = clSetKernelArg(kernel_upsampleDistortion, 0, sizeof(cl_mem), (void *)&return_predictionSignal_memObj);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 1, sizeof(cl_int), (void *)&frameWidth);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 2, sizeof(cl_int), (void *)&frameHeight);
    // Complete boundaries
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 3, sizeof(cl_mem), (void *)&refT_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 4, sizeof(cl_mem), (void *)&refL_64x64_memObj);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 5, sizeof(cl_mem), (void *)&refT_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 6, sizeof(cl_mem), (void *)&refL_32x32_memObj);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 7, sizeof(cl_mem), (void *)&refT_16x16_memObj);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 8, sizeof(cl_mem), (void *)&refL_16x16_memObj);
    // Reference samples and final distortion
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 9, sizeof(cl_mem), (void *)&return_SAD_memObj);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 10, sizeof(cl_mem), (void *)&return_SATD_memObj);
    error_1 |= clSetKernelArg(kernel_upsampleDistortion, 11, sizeof(cl_mem), (void *)&referenceFrame_memObj);

    probe_error(error_1, (char *)"Error setting arguments for the kernel\n");

    // Execute the OpenCL kernel on the list
    // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
    global_item_size = nWG * itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
    local_item_size = itemsPerWG;

    error = clEnqueueNDRangeKernel(command_queue, kernel_upsampleDistortion, 1, NULL,
                                   &global_item_size, &local_item_size, 0, NULL, &event);
    probe_error(error, (char *)"Error enqueuing kernel\n");

    error = clWaitForEvents(1, &event);
    probe_error(error, (char *)"Error waiting for events\n");

    error = clFinish(command_queue);
    probe_error(error, (char *)"Error finishing\n");

    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
    nanoSeconds = time_end - time_start;

    execTime_upsampleDistortion = nanoSeconds;

    readMemobjsIntoArray_Distortion(command_queue, nCTUs, PREDICTION_MODES_ID2*2, return_SAD_memObj, return_SAD, return_SATD_memObj, return_SATD);

    // REPORT DISTORTION VALUES TO TERMINAL
    if(reportToTerminal){
        if(reportDistortionOnlyTarget)
            reportTargetDistortionValues(return_SAD, return_SATD, nCTUs, targetCTU);
        else
            reportAllDistortionValues(return_SAD, return_SATD, nCTUs);
    }


    reportTimingResults();



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

    // REMOVE
    // Clean up
    error = clFlush(command_queue);
    error |= clFinish(command_queue);
    error |= clReleaseKernel(kernel_initRefSamples);
    error |= clReleaseKernel(kernel_reducedPrediction);
    error |= clReleaseProgram(program);
    error |= clReleaseCommandQueue(command_queue);
    error |= clReleaseMemObject(redT_64x64_memObj);
    error |= clReleaseMemObject(redL_64x64_memObj);
    error |= clReleaseMemObject(redT_32x32_memObj);
    error |= clReleaseMemObject(redL_32x32_memObj);
    error |= clReleaseMemObject(redT_16x16_memObj);
    error |= clReleaseMemObject(redL_16x16_memObj);
    error |= clReleaseMemObject(refT_64x64_memObj);
    error |= clReleaseMemObject(refL_64x64_memObj);
    error |= clReleaseMemObject(refT_32x32_memObj);
    error |= clReleaseMemObject(refL_32x32_memObj);
    error |= clReleaseMemObject(refT_16x16_memObj);
    error |= clReleaseMemObject(refL_16x16_memObj);
    error |= clReleaseMemObject(referenceFrame_memObj);
    error |= clReleaseMemObject(return_predictionSignal_memObj);
    error |= clReleaseMemObject(return_SATD_memObj);
    error |= clReleaseMemObject(return_SAD_memObj);
    error |= clReleaseMemObject(debug_mem_obj);
    probe_error(error, (char *)"Error releasing  OpenCL objects\n");

    free(source_str);
    free(platform_id);
    free(reference_frame);
    free(return_reducedPredictionSignal);
    free(return_SATD);
    free(return_SAD);
    free(return_redT_64x64);
    free(return_redL_64x64);
    free(return_redT_32x32);
    free(return_redL_32x32);
    free(return_redT_16x16);
    free(return_redL_16x16);
    free(return_refT_64x64);
    free(return_refL_64x64);
    free(return_refT_32x32);
    free(return_refL_32x32);
    free(return_refT_16x16);
    free(return_refL_16x16);
    free(debug_data);

    return 0;
}