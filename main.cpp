

#define USE_ARM 0


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120


#define USE_ALTERNATIVE_SAMPLES 1
#define PERFORM_CPU_FILTERING 1
#define ONLY_FILTER_AND_EXIT 0

#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <sstream> 
#include <fstream> 
#include <string.h>
#include <limits.h>
#include <assert.h>
#include "main_aux_functions.h"
#include <boost/program_options.hpp>


using namespace std;

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <CL/cl_ext.h>


#define MAX_SOURCE_SIZE (0x100000)

void callback(const char *buffer, size_t length, size_t final, void *user_data)
{
     fwrite(buffer, 1, length, stdout);
}

int main(int argc, char *argv[])
{


    int po_gpuDeviceIdx=-1, po_error=0, po_nFrames=-1, po_kernelIdx=-1;
    string po_outputPreffix, po_inputOrigFrames, po_resolution, po_filterType; 
    
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "produce help message")
    ("DeviceIndex",          po::value<int>(&po_gpuDeviceIdx)->default_value(0),      "Index of the GPU device according ot clinfo command")
    ("FramesToBeEncoded,f",  po::value<int>(&po_nFrames),                             "Number of frames to be processed")
    ("Resolution,s",         po::value<string>(&po_resolution),                       "Resolution of the video, in the format 1920x1080")
    ("OriginalFrames,o",     po::value<string>(&po_inputOrigFrames),                  "Input file for original frames samples")
    ("OutputPreffix,l",      po::value<string>(&po_outputPreffix)->default_value(""), "Output files preffix with produced CPMVs")
    ("FilterType",           po::value<string>(&po_filterType),                       "Type of smoothing filter"  )
    ("KernelIdx",            po::value<int>(&po_kernelIdx)->default_value(0),         "Index of the filtering kernel used to define the coeffiients")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm); 
    if (vm.count("help")) {
        cout << desc << "\n";
        return 1;
    }

    po_error = checkReportParameters(vm);

    int isAvail = USE_ARM ? isFilterAvailable_arm(po_filterType) : isFilterAvailable(po_filterType);

    if( ! isAvail){
        cout << "  [!] ERROR: Filter type " << po_filterType << " not supported" << endl;
        exit(0);
    }

    if(po_error > 0){
        cout << "Exiting after finding errors in input parameters" << endl;
        return 1;
    }



    if(TRACE_POWER)
        print_timestamp((char*)"STARTED HOST");

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
 
    #if USE_ARM
        fp = fopen("intra_arm.cl", "r");
    #else
        fp = fopen("intra.cl", "r");
    #endif

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

    if(!TRACE_POWER)
        printf("\n");

    for (cl_uint p = 0; p < ret_num_platforms; p++)
    {
        error = clGetPlatformInfo(platform_id[p], CL_PLATFORM_NAME, 128 * sizeof(char), platform_name, NULL);
        probe_error(error, (char*)"Error querying CL_PLATFORM_NAME\n");
        if(!TRACE_POWER)
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
    if(!TRACE_POWER)
        printf("\n");

    char device_name[1024];
    char device_extensions[1024];

    if(!TRACE_POWER){
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
    }

    // Create "target" device and assign proper IDs
    cl_device_id device_id = NULL; 
    
    // Select what CPU or GPU will be used based on parameters
    if( po_gpuDeviceIdx < assigned_gpus){
        cout << "COMPUTING ON GPU " << po_gpuDeviceIdx << endl;        
        device_id = gpu_device_ids[po_gpuDeviceIdx];    
    }
    else{
        cout << "Incorrect GPU index. Only " << assigned_gpus << " GPUs are detected" << endl;
        exit(0); 
    }

    size_t ret_val;
    cl_uint max_compute_units;
    error = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, 0, NULL, &ret_val);
    error|= clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, ret_val, &max_compute_units, NULL);
    probe_error(error, (char*)"Error querying maximum number of compute units of device\n");
    if(!TRACE_POWER)
        cout << "-- Max compute units " << max_compute_units << endl;


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////         STARTS BY CREATING A CONTEXT, QUEUE, AND MOVING DATA INTO THE BUFFERS         /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Create an OpenCL context
    
     cl_context_properties context_properties[] =
     { 
          CL_CONTEXT_PLATFORM, 0,
          CL_PRINTF_CALLBACK_ARM, (cl_context_properties)callback,
          CL_PRINTF_BUFFERSIZE_ARM, 0x1000,
          0
     };

    context_properties[1] = (cl_context_properties)platform_id[0];

    #if USE_ARM
        cl_context context = clCreateContext( context_properties, 1, &device_id, NULL, NULL, &error);
    #else
        cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &error);
    #endif
    
    probe_error(error, (char*)"Error creating context\n");

    // Create a command queue
    // Profiling enabled to measure execution time. 
    // TODO: Remove this profiling when perform actual computation, it may slowdown the processing (for experiments and etc)

    // "common" queue is shared for writing and kernels that deal with all SizeIds.
    // "idX" queues deal with distortion of specific SizeIds. 
    // "read" queue deals with reading distortion WITHOUT BLOCKING

    cl_command_queue command_queue_write = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    cl_command_queue command_queue_common = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    cl_command_queue command_queue_id2 = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    cl_command_queue command_queue_id1 = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    cl_command_queue command_queue_id0 = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    cl_command_queue command_queue_read = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &error);
    probe_error(error, (char*)"Error creating command queue\n");

    stringstream res(po_resolution);
    vector<string> tokens;
    char split_char = 'x';

    for(string each; getline(res, each, split_char); tokens.push_back(each));
    if(tokens.size() != 2){
        cout << "  [!] ERROR: Input resolution \"" << po_resolution << "\" not set properly" << endl;
        return 0;
    }
    
    const int frameWidth  = stoi(tokens[0]); // 1920; // 1920 or 3840
    const int frameHeight = stoi(tokens[1]); // 1080; // 1080 or 2160
    const int nCtus = getNumCtus(frameWidth, frameHeight); // frameHeight==1080 ? 135 : 510; //135 or 510 for 1080p and 2160p  ||  1080p videos have 120 entire CTUs plus 15 partial CTUs || 4k videos have 480 entire CTUs plus 30 partial CTUs
    if( nCtus == 0){
        printf("[!] ERROR: Unsupported resolution %dx%d\n", frameWidth, frameHeight);
        printf("Supported resolutions are:\n");
        for(unsigned int i=0; i<availableRes.size(); i++){
            printf("  %dx%d\n", get<0>(availableRes[i]), get<1>(availableRes[i]) );
        }
        return 0;
    }

    const int itemsPerWG_obtainReducedBoundaries = 128;
    const int itemsPerWG_obtainReducedPrediction = 256;
    const int itemsPerWG_upsampleDistortion = 256;

    int itemsPerWG;
    int nWG; // All CU sizes inside all CTUs are being processed simultaneously by distinct WGs

    // Read the input data
    string refFrameFileName = po_inputOrigFrames; // argv[3];

    string outputFilePreffix = po_outputPreffix; // argv[4]; // Preffix of exported files containing the prediction signal

    N_FRAMES = po_nFrames; // stoi(argv[5]);

    ifstream refFile;
    refFile.open(refFrameFileName);

    if (!refFile.is_open())
    { // validate file open for reading
        perror(("error while opening samples files"));
        return 1;
    }

    string refLine, refVal, currLine, currVal;

    const int FRAME_SIZE = frameWidth*frameHeight;
    // const int nCTUs = frameHeight==1080 ? 135 : 510; //135 or 510 for 1080p and 2160p  ||  1080p videos have 120 entire CTUs plus 15 partial CTUs || 4k videos have 480 entire CTUs plus 30 partial CTUs
    int nCTUs;
    switch(frameHeight){
        case 240:
            nCTUs = 8;
            break;
        case 480:
            nCTUs = 28;
            break;
        case 720:
            nCTUs = 60;
            break;
        case 1080:
            nCTUs = 135;
            break;
        case 2160:
            nCTUs = 510;
            break;
        default:
            printf("ERROR COMPUTING NUMBER OF CTUS\n");
            return 0;
    }

    const int NUM_PRED_MODES = PREDICTION_MODES_ID2;
    const int TEST_TRANSPOSED_MODES = 1;
    const int TOTAL_PREDICTION_MODES = NUM_PRED_MODES * (TEST_TRANSPOSED_MODES ? 2 : 1);

    unsigned short *reference_frame = (unsigned short *)malloc(sizeof(short) * FRAME_SIZE * N_FRAMES);

    if(TRACE_POWER)
        print_timestamp((char*) "START READ SAMPLES .csv");

    // Read the samples from reference frame into the reference array
    for(int f=0; f<N_FRAMES; f++)
    {
        for (int h=0; h<frameHeight; h++)
        {
            getline(refFile, refLine, '\n');
            stringstream currStream(currLine), refStream(refLine); 

            for (int w = 0; w < frameWidth; w++)
            {
                getline(currStream, currVal, ',');
                getline(refStream, refVal, ',');
                reference_frame[f*frameWidth*frameHeight + h*frameWidth + w] = stoi(refVal);
            }
        }
    }

    if(TRACE_POWER)
        print_timestamp((char*) "FINISH READ SAMPLES .csv");


#if USE_ALTERNATIVE_SAMPLES || PERFORM_CPU_FILTERING
    int kernelIdx = po_kernelIdx ;
#endif


#if PERFORM_CPU_FILTERING
    int maxThreads = 256;
    int reportFilterResults = 0;
    
    int kernelDim = 3;
    profileCpuFiltering(reference_frame, frameWidth, frameHeight, kernelDim, kernelIdx, maxThreads, reportFilterResults);

    kernelDim = 5;
    profileCpuFiltering(reference_frame, frameWidth, frameHeight, kernelDim, kernelIdx, maxThreads, reportFilterResults);

    // return 1;
#endif

    // These buffers are used to store the reduced boundaries for all CU sizes
    error = 0;


    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////              THESE MEMORY OBJECTS AND ARRAYS MUST BE FREED AFTER EXECUTION            /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // These memory objects are used to store intermediate data and debugging information from the kernel
    cl_mem debug_mem_obj;

    // Used for all sizeId=2 CU sizes together
    cl_mem redT_all_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              BUFFER_SLOTS * nCTUs * (ALL_TOTAL_CUS_SizeId12_PER_CTU * BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU * BOUNDARY_SIZE_Id0) * sizeof(short), NULL, &error_1);    
    cl_mem redL_all_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              BUFFER_SLOTS * nCTUs * (ALL_TOTAL_CUS_SizeId12_PER_CTU * BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU * BOUNDARY_SIZE_Id0) * sizeof(short), NULL, &error_2);    
    
    
    cl_mem refT_all_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              BUFFER_SLOTS * nCTUs * ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES] * sizeof(short), NULL, &error_3);
    cl_mem refL_all_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                              BUFFER_SLOTS * nCTUs * ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES] * sizeof(short), NULL, &error_4);

    error = error_1 || error_2 || error_3 || error_4;

    
    cl_mem referenceFrame_memObj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                  N_FRAMES * FRAME_SIZE * sizeof(short), NULL, &error_1);

    cl_mem filteredFrame_memObj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                                  N_FRAMES * FRAME_SIZE * sizeof(short), NULL, &error_2);

    error = error || error_1 || error_2;

    // These memory objects hold the predicted signal and distortion after the kernel has finished
    cl_mem return_predictionSignal_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                                           BUFFER_SLOTS * nCTUs * ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(short), NULL, &error_1); // Each CTU is composed of TOTAL_CUS_PER_CTU CUs, and each reduced CU is 8*8, and we have 12 prediction modes

    cl_mem return_SATD_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                               BUFFER_SLOTS * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(long), NULL, &error_2);
    cl_mem return_SAD_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                               BUFFER_SLOTS * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(long), NULL, &error_3);


    cl_mem return_minSadHad_memObj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                               BUFFER_SLOTS * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES] * sizeof(long), NULL, &error_4);

    error = error || error_1 || error_2 || error_3 || error_4;

    probe_error(error, (char*)"Error creating memory buffers\n");

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////       CREATE A PROGRAM (OBJECT) BASED ON .cl FILE AND BUILD IT TO TARGET DEVICE       /////
    /////         CREATE A KERNEL BY ASSIGNING A NAME FOR THE RECENTLY COMPILED PROGRAM         /////
    /////                           LOADS THE ARGUMENTS FOR THE KERNEL                          /////
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Create a program from the kernel source (specifically for the device in the context variable)
    cl_program program_sizeid2, program_sizeid1, program_sizeid0; // Used to compile the upsample kernel with difference constants and use the same code
    
    program_sizeid2 = clCreateProgramWithSource(context, 1, 
                                                   (const char **)&source_str, (const size_t *)&source_size, &error);
    probe_error(error, (char*)"Error creating program from source with SizeId=2\n");
    
    program_sizeid1 = clCreateProgramWithSource(context, 1, 
                                                   (const char **)&source_str, (const size_t *)&source_size, &error);
    probe_error(error, (char*)"Error creating program from source with SizeId=1\n");
    
    program_sizeid0 = clCreateProgramWithSource(context, 1, 
                                                   (const char **)&source_str, (const size_t *)&source_size, &error);
    probe_error(error, (char*)"Error creating program from source with SizeId=0\n");

    // Build the program
    char buildOptions2[100], buildOptions1[100], buildOptions0[100];

    if(TRACE_POWER)
        print_timestamp((char*) "START BUILD KERNELS");

    sprintf(buildOptions2, "-DSIZEID=%d -DTRACE_POWER=%d -DN_FRAMES=%d -DMAX_PERFORMANCE_DIST=%d", 2, TRACE_POWER, N_FRAMES, MAX_PERFORMANCE_DIST);
    error = clBuildProgram(program_sizeid2, 1, &device_id, buildOptions2, NULL, NULL);
    probe_error(error, (char*)"Error building the program with SizeId=2\n");
    // Show debugging information when the build is not successful
    if (error == CL_BUILD_PROGRAM_FAILURE)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_sizeid2, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program_sizeid2, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
    }


    sprintf(buildOptions1, "-DSIZEID=%d -DTRACE_POWER=%d -DN_FRAMES=%d -DMAX_PERFORMANCE_DIST=%d", 1, TRACE_POWER, N_FRAMES, MAX_PERFORMANCE_DIST);
    error = clBuildProgram(program_sizeid1, 1, &device_id, buildOptions1, NULL, NULL);
    probe_error(error, (char*)"Error building the program with SizeId=1\n");
    // Show debugging information when the build is not successful
    if (error == CL_BUILD_PROGRAM_FAILURE)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_sizeid1, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program_sizeid1, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
    }


    sprintf(buildOptions0, "-DSIZEID=%d -DTRACE_POWER=%d -DN_FRAMES=%d -DMAX_PERFORMANCE_DIST=%d", 0, TRACE_POWER, N_FRAMES, MAX_PERFORMANCE_DIST);
    error = clBuildProgram(program_sizeid0, 1, &device_id, buildOptions0, NULL, NULL);
    probe_error(error, (char*)"Error building the program with SizeId=0\n");
    // Show debugging information when the build is not successful
    if (error == CL_BUILD_PROGRAM_FAILURE)
    {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(program_sizeid0, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        // Allocate memory for the log
        char *log = (char *)malloc(log_size);

        // Get the log
        clGetProgramBuildInfo(program_sizeid0, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

        // Print the log
        printf("%s\n", log);
        free(log);
    }
    if(TRACE_POWER)
        print_timestamp((char*) "FINISH BUILD KERNELS");


    ////////////////////////////////////////////////////////////////
    /////                                                      /////
    /////       WRITE REFERENCE SAMPLES INTO GPU MEMORY        /////
    /////                                                      /////
    ////////////////////////////////////////////////////////////////

    double nanoSeconds = 0;
    // These variabels are used to profile the time spend writing to memory objects "clEnqueueWriteBuffer"
    cl_ulong write_time_start;
    cl_ulong write_time_end;
    cl_event write_event;

    if(TRACE_POWER){
        print_timestamp((char*) "START WRITE SAMPLES MEMOBJ");
        save_startTime();
    }
        
    
    
    /*
    
    ONLY WRITE THE FIRST FRAME NOW
    THE NEXT FRAMES WILL BE WRITTEN ON DEMAND INSIDE THE FOR LOOP REUSING MEMORY SLOTS (CIRCULAR BUFFER)

    */

   error = clEnqueueWriteBuffer(command_queue_write, referenceFrame_memObj, CL_TRUE, 0,
                                1 * FRAME_SIZE * sizeof(short), reference_frame, 0, NULL, &write_event);
    // error = clWaitForEvents(1, &write_event);
    // probe_error(error, (char*)"Error waiting for write events\n");  
    error = clFinish(command_queue_write);
    probe_error(error, (char*)"Error finishing write\n");
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(write_time_start), &write_time_start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(write_time_end), &write_time_end, NULL);
    nanoSeconds += write_time_end-write_time_start;

    if(TRACE_POWER)
        print_timestamp((char*) "FINISH WRITE SAMPLES MEMOBJ");
    

    writeTime = nanoSeconds;
    writeTime_filter = nanoSeconds;
    nanoSeconds = 0.0;

    probe_error(error, (char*)"Error copying data from memory to buffers LEGACY\n");

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /////                  VARIABLES SHARED BETWEEN THE EXECUTION OF ALL KERNELS                /////
    /////           THIS INCLUDES CONSTANTS, VARIABLES USED FOR CONTROL AND PROFILING           /////
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Used to access optimal workgroup size
    size_t size_ret;
    cl_uint preferred_size, maximum_size;

    cl_kernel kernel_filterFrames, kernel_initRefSamples, kernel_reducedPrediction, kernel_upsampleDistortion; // The object that holds the compiled kernel to be enqueued

    // Used to profile execution time of kernel
    cl_event event;
    cl_ulong time_start, time_end;

    // Used to set the workgroup sizes
    size_t global_item_size, local_item_size;

    // Used to export the kernel results into proper files or the terminal
    string exportFileName;

    int enableTerminalReport = 0;
    int reportReducedBoundaries = 1;
    int reportCompleteBoundaries = 1;
    int reportReducedPrediction = 1;
    int reportDistortion = 1;
    int reportDistortionOnlyTarget = 1;

    int reportDistortionToFile = 1;
    int targetCTU = 16;

    ///////////////////////////////////////////////////////////////////////////////////////
    /////              THESE DYNAMIC ARRAYS  MUST BE FREED AFTER EXECUTION            /////
    ///////////////////////////////////////////////////////////////////////////////////////

    // These dynamic arrays retrieve the information from the kernel to the host
    // Predicted signal and distortion
    
    short *return_filteredFrame;
    short *return_reducedPredictionSignal;
    long *return_SATD, *return_SAD, *return_minSadHad;
    short *return_unified_redT, *return_unified_redL;
    short *return_unified_refT, *return_unified_refL;

    // Debug information returned by kernel
    short *debug_data;

    // TODO: Correct to the proper number of WGs and CTUs
    // Currently, each WG will process one CTU partitioned into one CU size, and produce the reduced boundaries for each one of the CUs
    nWG = nCTUs*ALL_NUM_CU_SIZES;
    
    debug_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   nWG * itemsPerWG_upsampleDistortion * 4 * sizeof(cl_short), NULL, &error_1);

    // -----------------------------
    // Allocate some memory space
    // -----------------------------
    return_filteredFrame = (short*)malloc(sizeof(short) * N_FRAMES * FRAME_SIZE);
    return_reducedPredictionSignal = (short*)malloc(sizeof(short) * N_FRAMES * nCTUs * ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES]); // Each predicted CU has 8x8 samples
    return_SATD = (long*) malloc(sizeof(long) * N_FRAMES * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES]);
    return_SAD = (long*) malloc(sizeof(long) * N_FRAMES * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES]);
    return_minSadHad = (long*) malloc(sizeof(long) * N_FRAMES * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES]);   
    // Unified boundaries
    return_unified_redT = (short*) malloc(sizeof(short) * N_FRAMES * nCTUs * (ALL_TOTAL_CUS_SizeId12_PER_CTU * BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU * BOUNDARY_SIZE_Id0));
    return_unified_redL = (short*) malloc(sizeof(short) * N_FRAMES * nCTUs * (ALL_TOTAL_CUS_SizeId12_PER_CTU * BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU * BOUNDARY_SIZE_Id0));
    return_unified_refT = (short*) malloc(sizeof(short) * N_FRAMES * nCTUs * ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES]);
    return_unified_refL = (short*) malloc(sizeof(short) * N_FRAMES * nCTUs * ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES]);

    // Debug information returned by kernel
    debug_data = (short*) malloc(sizeof(short) * nWG*itemsPerWG_upsampleDistortion*4);

    // These memory objects are used to store intermediate data and debugging information from the kernel
    debug_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                   nWG * itemsPerWG_upsampleDistortion * 4 * sizeof(cl_short), NULL, &error_1);

    error = error_1;
    probe_error(error,(char*)"Error creating memory object for debugging information\n");


    for(cl_int curr=0; curr < N_FRAMES; curr++){

        cl_int currFrame = curr;
        printf("Current frame %d\n", curr);


#if USE_ALTERNATIVE_SAMPLES

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //          FILTER THE ORIGINAL INPUT SAMPLES
        //
        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        #if USE_ARM
            itemsPerWG = 128;
            int nWG_Filter = nCTUs*8; // Adjust based on the filtering kernel. nCTUs*2 or nCTUs*4
        #else
            itemsPerWG = 256;
            int nWG_Filter = nCTUs*4; // Adjust based on the filtering kernel. nCTUs*2 or nCTUs*4
        #endif

        

        // Create kernel
        
        //                                    When using filterFrame_2d, int or float, nWG_Filter MUST BE nCTUs*2
        //                                       When using filterFrame_1d_float, nWG_Filter MUST BE nCTUs*4
        //                                                             |
        kernel_filterFrames = clCreateKernel(program_sizeid2, po_filterType.c_str(), &error);
        probe_error(error, (char*)"Error creating filterFrame kernel\n"); 
        printf("Performing filterFrame kernel...\n");

        // Query for work groups sizes information
        error = clGetKernelWorkGroupInfo(kernel_filterFrames, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_filterFrames, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
        error |= clGetKernelWorkGroupInfo(kernel_filterFrames, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_filterFrames, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

        probe_error(error, (char*)"Error querying preferred or maximum work group size\n");
        //cout << "-- Preferred WG size multiple " << preferred_size << endl;
        //cout << "-- Maximum WG size " << maximum_size << endl;

        currFrame = curr;
        // Set the arguments of the kernel initBoundaries
        error_1 = clSetKernelArg(kernel_filterFrames, 0, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        error_1 = clSetKernelArg(kernel_filterFrames, 1, sizeof(cl_mem), (void *)&filteredFrame_memObj);
        error_1 |= clSetKernelArg(kernel_filterFrames, 2, sizeof(cl_int), (void *)&frameWidth);
        error_1 |= clSetKernelArg(kernel_filterFrames, 3, sizeof(cl_int), (void *)&frameHeight);
        error_1 |= clSetKernelArg(kernel_filterFrames, 4, sizeof(cl_int), (void *)&kernelIdx);
        error_1 |= clSetKernelArg(kernel_filterFrames, 5, sizeof(cl_int), (void *)&currFrame);


        probe_error(error_1, (char*)"Error setting arguments for the kernel\n");

        // Execute the OpenCL kernel on the list
        // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
        global_item_size = nWG_Filter*itemsPerWG; // frameWidth*frameHeight; // nWG_Filter*itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
        local_item_size = itemsPerWG;

        if(TRACE_POWER)
            print_timestamp((char*) "START ENQUEUE filterFrame");

        error = clEnqueueNDRangeKernel(command_queue_common, kernel_filterFrames, 1, NULL,
                                    &global_item_size, &local_item_size, 0, NULL, &event);
        probe_error(error, (char*)"Error enqueuing kernel\n");
        error = clWaitForEvents(1, &event);
        probe_error(error, (char*)"Error waiting for events\n");

        error = clFinish(command_queue_common);
        probe_error(error, (char*)"Error finishing filterSamples\n");

        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        nanoSeconds = time_end-time_start;

        printf("FilterSamples took %f ms\n\n", nanoSeconds/1000000);
        executionTime_filter = nanoSeconds;
        nanoSeconds = 0;
        


        if(TRACE_POWER)
            print_timestamp((char*) "FINISH ENQUEUE filterFrame");


        // execTime += execTime_reducedBoundaries;
        
        if(enableTerminalReport){
            // Read filtering results from memory objects into host arrays
            readMemobjsIntoArray_FilteredFrame(command_queue_common, frameWidth, frameHeight, filteredFrame_memObj, return_filteredFrame);
        }

        printf("TIMING REPORT\n");
        printf("Write(ns): %f\n", writeTime_filter);
        printf("Execution(ns):%f\n", executionTime_filter);
        printf("Read(ns): %f\n", readTime_filter);
        printf("TotalFilterTime(ms): %f\n", (writeTime_filter+executionTime_filter+readTime_filter)/1000000 );

        // printf("\n\nGPU FILTERED SAMPLES\n\n");

        // for(int h=0; h<frameHeight; h++){
        //     for(int w=0; w<frameWidth; w++){
        //         // reference_frame
        //         // return_filteredFrame
        //         printf("%hd,", return_filteredFrame[h*frameWidth+w]);
        //     }
        //     printf("\n");
        // }
#if ONLY_FILTER_AND_EXIT        
    return 1;
#endif

#endif

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //          FIRST WE MUST OBTAIN THE REDUCED BOUNDARIES
        //
        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        itemsPerWG = itemsPerWG_obtainReducedBoundaries;

        // Create kernel
        kernel_initRefSamples = clCreateKernel(program_sizeid2, "initBoundaries", &error);
        probe_error(error, (char*)"Error creating initBoundaries kernel\n"); 
        printf("Performing initBoundaries kernel...\n");

        // Query for work groups sizes information
        error = clGetKernelWorkGroupInfo(kernel_initRefSamples, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_initRefSamples, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
        error |= clGetKernelWorkGroupInfo(kernel_initRefSamples, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_initRefSamples, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

        probe_error(error, (char*)"Error querying preferred or maximum work group size\n");
        //cout << "-- Preferred WG size multiple " << preferred_size << endl;
        //cout << "-- Maximum WG size " << maximum_size << endl;

        currFrame = curr;
        // Set the arguments of the kernel initBoundaries
        #if USE_ALTERNATIVE_SAMPLES
            error_1 = clSetKernelArg(kernel_initRefSamples, 0, sizeof(cl_mem), (void *)&filteredFrame_memObj);
        #else
            error_1 = clSetKernelArg(kernel_initRefSamples, 0, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        #endif
        error_1 |= clSetKernelArg(kernel_initRefSamples, 1, sizeof(cl_int), (void *)&frameWidth);
        error_1 |= clSetKernelArg(kernel_initRefSamples, 2, sizeof(cl_int), (void *)&frameHeight);
        // Combined reduced and complete boundaries for all CU
        error_1 |= clSetKernelArg(kernel_initRefSamples, 3, sizeof(cl_mem), (void *)&redT_all_memObj);
        error_1 |= clSetKernelArg(kernel_initRefSamples, 4, sizeof(cl_mem), (void *)&redL_all_memObj);
        error_1 |= clSetKernelArg(kernel_initRefSamples, 5, sizeof(cl_mem), (void *)&refT_all_memObj);
        error_1 |= clSetKernelArg(kernel_initRefSamples, 6, sizeof(cl_mem), (void *)&refL_all_memObj);
        error_1 |= clSetKernelArg(kernel_initRefSamples, 7, sizeof(cl_int), (void *)&currFrame);


        probe_error(error_1, (char*)"Error setting arguments for the kernel\n");

        // Execute the OpenCL kernel on the list
        // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
        global_item_size = nWG*itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
        local_item_size = itemsPerWG;

        if(TRACE_POWER)
            print_timestamp((char*) "START ENQUEUE initBoundaries");

        error = clEnqueueNDRangeKernel(command_queue_common, kernel_initRefSamples, 1, NULL,
                                    &global_item_size, &local_item_size, 0, NULL, &event);
        probe_error(error, (char*)"Error enqueuing kernel\n");

        //error = clWaitForEvents(1, &event);
        //probe_error(error, (char*)"Error waiting for events\n");

        //error = clFinish(command_queue_common);
        //probe_error(error, (char*)"Error finishing\n");

        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        //nanoSeconds = time_end-time_start;

        //execTime_reducedBoundaries += nanoSeconds;

        // nanoSeconds = 0;

        if(TRACE_POWER)
            print_timestamp((char*) "FINISH ENQUEUE initBoundaries");

        // execTime += execTime_reducedBoundaries;
        
        // ----------------  EXPORT BOUNDARIES FORM UNIFIED BUFFERS
        //
        if(enableTerminalReport){
            // Read affine results from memory objects into host arrays
            readMemobjsIntoArray_UnifiedBoundaries(command_queue_common, nCTUs, redT_all_memObj, refT_all_memObj, return_unified_redT, return_unified_refT, redL_all_memObj, refL_all_memObj, return_unified_redL, return_unified_refL);
        
            if(reportReducedBoundaries)
                reportReducedBoundariesTargetCtu_ALL(return_unified_redT, return_unified_redL, targetCTU, frameWidth, frameHeight);
            if(reportCompleteBoundaries)
                reportCompleteBoundariesTargetCtu_ALL(return_unified_refT, return_unified_refL, targetCTU, frameWidth, frameHeight);
        }
        
        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //          WHILE THE BOUNDARIES ARE BEING INITIALIZED, WE CAN START COPYING THE NEXT FRAME FROM HOST INTO GPU USING A NON-BLOCKING COPY
        //          WRITING ONE FRAME IS MUCH FASER THAN DOING UPSAMPLING SO THERE SHOULD BE NO NEED ADD SYNCH BARRIERS HERE    
        //
        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        // Only copy the number of available frames. In the next iteration we will not copy anything    
        if(curr+1<N_FRAMES){
            if(TRACE_POWER){
                print_timestamp((char*) "START WRITE SAMPLES MEMOBJ");
            }

            error = clFinish(command_queue_write);
            probe_error(error, (char*)"Error finishing write\n");
                                                                                    // non-blocking
                                                                                            // offset in mem_obj
            error = clEnqueueWriteBuffer(command_queue_write, referenceFrame_memObj, CL_FALSE, ((curr+1)%BUFFER_SLOTS) * FRAME_SIZE * sizeof(short),
                                                                        // c++ array with offset
                                        1 * FRAME_SIZE * sizeof(short), reference_frame+(curr+1)*FRAME_SIZE, 0, NULL, &write_event);
        }
        
        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //          NOW WE OBTAIN THE REDUCED PREDICTION FOR ALL CU SIZES AND PREDICTION MODES
        //
        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        itemsPerWG = itemsPerWG_obtainReducedPrediction;

        // Create kernel
        kernel_reducedPrediction = clCreateKernel(program_sizeid2, "MIP_ReducedPred", &error);
        probe_error(error, (char *)"Error creating MIP_ReducedPred kernel\n");
        printf("Performing MIP_ReducedPred kernel...\n");

        // Query for work groups sizes information
        error = clGetKernelWorkGroupInfo(kernel_reducedPrediction, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_reducedPrediction, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
        error |= clGetKernelWorkGroupInfo(kernel_reducedPrediction, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_reducedPrediction, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

        probe_error(error, (char *)"Error querying preferred or maximum work group size\n");
        //cout << "-- Preferred WG size multiple " << preferred_size << endl;
        //cout << "-- Maximum WG size " << maximum_size << endl;

        currFrame = curr;
        // Set the arguments of the MIP_ReducedPred kernel
        error_1  = clSetKernelArg(kernel_reducedPrediction, 0, sizeof(cl_mem), (void *)&return_predictionSignal_memObj);
        error_1 |= clSetKernelArg(kernel_reducedPrediction, 1, sizeof(cl_int), (void *)&frameWidth);
        error_1 |= clSetKernelArg(kernel_reducedPrediction, 2, sizeof(cl_int), (void *)&frameHeight);
        error_1 |= clSetKernelArg(kernel_reducedPrediction, 3, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        // Unified reduced boundariers
        error_1 |= clSetKernelArg(kernel_reducedPrediction, 4, sizeof(cl_mem), (void *)&redT_all_memObj);
        error_1 |= clSetKernelArg(kernel_reducedPrediction, 5, sizeof(cl_mem), (void *)&redL_all_memObj);
        error_1 |= clSetKernelArg(kernel_reducedPrediction, 6, sizeof(cl_int), (void *)&currFrame);

        probe_error(error_1, (char *)"Error setting arguments for the kernel\n");

        // Execute the OpenCL kernel on the list
        // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
        global_item_size = nWG * itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
        local_item_size = itemsPerWG;

        if(TRACE_POWER)
            print_timestamp((char*) "START ENQUEUE reducedPred");


        error = clEnqueueNDRangeKernel(command_queue_common, kernel_reducedPrediction, 1, NULL,
                                        &global_item_size, &local_item_size, 0, NULL, &event);
        probe_error(error, (char *)"Error enqueuing kernel\n");

        //error = clWaitForEvents(1, &event);
        //probe_error(error, (char *)"Error waiting for events\n");

        //error = clFinish(command_queue_common);
        //probe_error(error, (char *)"Error finishing\n");


        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        //nanoSeconds = time_end - time_start;

        //execTime_reducedPrediction += nanoSeconds;


        if(TRACE_POWER)
            print_timestamp((char*) "FINISH ENQUEUE reducedPred");

        // execTime += execTime_reducedPrediction;
            

        if(enableTerminalReport){
            readMemobjsIntoArray_reducedPrediction(command_queue_common, nCTUs, TOTAL_PREDICTION_MODES, return_predictionSignal_memObj,  return_reducedPredictionSignal, return_SAD_memObj, return_SAD);
            if(reportReducedPrediction)
                reportReducedPredictionTargetCtu_ALL(return_reducedPredictionSignal, targetCTU, frameWidth, frameHeight);
        }
            

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //          FINALLY, UPSAMPLE THE REDUCED PREDICTION AND COMPUTE THE DISTOTION
        //
        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        itemsPerWG = itemsPerWG_upsampleDistortion;

        // We must wait until all CUs are predicted to compute the distortion
        // Since we use three queues for different SizeIds, an explicit synchronization is required
        error = clFinish(command_queue_common);
        probe_error(error, (char *)"Error finishing\n");


        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //          START WITH CUs OF SizeID=2

        // Create kernel
        kernel_upsampleDistortion = clCreateKernel(program_sizeid2, "upsampleDistortion", &error);
        probe_error(error, (char *)"Error creating upsampleDistortion kernel\n");
        printf("Performing upsampleDistortion kernel...\n");

        // Query for work groups sizes information
        error = clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

        probe_error(error, (char *)"Error querying preferred or maximum work group size\n");
        //cout << "-- Preferred WG size multiple " << preferred_size << endl;
        //cout << "-- Maximum WG size " << maximum_size << endl;

        currFrame = curr;
        // Set the arguments of the upsampleDistortion kernel
        error_1  = clSetKernelArg(kernel_upsampleDistortion, 0, sizeof(cl_mem), (void *)&return_predictionSignal_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 1, sizeof(cl_int), (void *)&frameWidth);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 2, sizeof(cl_int), (void *)&frameHeight);
        // Reference samples and final distortion
    #if MAX_PERFORMANCE_DIST // Only one distortion value is returned
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 3, sizeof(cl_mem), (void *)&return_minSadHad_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 4, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        // Unified boundariers
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 5, sizeof(cl_mem), (void *)&refT_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 6, sizeof(cl_mem), (void *)&refL_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 7, sizeof(cl_int), (void *)&currFrame);
    #else
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 3, sizeof(cl_mem), (void *)&return_SAD_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 4, sizeof(cl_mem), (void *)&return_SATD_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 5, sizeof(cl_mem), (void *)&return_minSadHad_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 6, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        // Unified boundariers
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 7, sizeof(cl_mem), (void *)&refT_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 8, sizeof(cl_mem), (void *)&refL_all_memObj);
    #endif
        
        probe_error(error_1, (char *)"Error setting arguments for the kernel\n");

        // Execute the OpenCL kernel on the list
        // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
        nWG = nCTUs*NUM_CU_SIZES_SizeId2;
        global_item_size = nWG * itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
        local_item_size = itemsPerWG;

        if(TRACE_POWER)
            print_timestamp((char*) "START ENQUEUE upsamplePred_SIZEID=2");


        error = clEnqueueNDRangeKernel(command_queue_id2, kernel_upsampleDistortion, 1, NULL,
                                        &global_item_size, &local_item_size, 0, NULL, &event);
        probe_error(error, (char *)"Error enqueuing kernel\n");

        //error = clWaitForEvents(1, &event);
        //probe_error(error, (char *)"Error waiting for events\n");

        //error = clFinish(command_queue_id2);
        //probe_error(error, (char *)"Error finishing\n");


        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        //nanoSeconds = time_end - time_start;

        //execTime_upsampleDistortion_SizeId2 += nanoSeconds;


        if(TRACE_POWER)
            print_timestamp((char*) "FINISH ENQUEUE upsamplePred_SIZEID=2");

        // execTime_upsampleDistortion += execTime_upsampleDistortion_SizeId2;
            


        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //          PROCEED TO CUs OF SizeID=1

        // Create kernel
        kernel_upsampleDistortion = clCreateKernel(program_sizeid1, "upsampleDistortion", &error);
        probe_error(error, (char *)"Error creating upsampleDistortion kernel\n");
        printf("Performing upsampleDistortion kernel...\n");

        // Query for work groups sizes information
        error = clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

        probe_error(error, (char *)"Error querying preferred or maximum work group size\n");
        //cout << "-- Preferred WG size multiple " << preferred_size << endl;
        //cout << "-- Maximum WG size " << maximum_size << endl;

        currFrame = curr;
        // Set the arguments of the upsampleDistortion kernel
        error_1  = clSetKernelArg(kernel_upsampleDistortion, 0, sizeof(cl_mem), (void *)&return_predictionSignal_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 1, sizeof(cl_int), (void *)&frameWidth);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 2, sizeof(cl_int), (void *)&frameHeight);
        // Reference samples and final distortion
    #if MAX_PERFORMANCE_DIST // Only one distortion value is returned
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 3, sizeof(cl_mem), (void *)&return_minSadHad_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 4, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        // Unified boundariers
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 5, sizeof(cl_mem), (void *)&refT_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 6, sizeof(cl_mem), (void *)&refL_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 7, sizeof(cl_int), (void *)&currFrame);
    #else
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 3, sizeof(cl_mem), (void *)&return_SAD_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 4, sizeof(cl_mem), (void *)&return_SATD_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 5, sizeof(cl_mem), (void *)&return_minSadHad_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 6, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        // Unified boundariers
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 7, sizeof(cl_mem), (void *)&refT_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 8, sizeof(cl_mem), (void *)&refL_all_memObj);
    #endif

        probe_error(error_1, (char *)"Error setting arguments for the kernel\n");

        // Execute the OpenCL kernel on the list
        // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
        nWG = nCTUs*NUM_CU_SIZES_SizeId1;
        global_item_size = nWG * itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
        local_item_size = itemsPerWG;

        if(TRACE_POWER)
            print_timestamp((char*) "START ENQUEUE upsamplePred_SIZEID=1");


        error = clEnqueueNDRangeKernel(command_queue_id1, kernel_upsampleDistortion, 1, NULL,
                                        &global_item_size, &local_item_size, 0, NULL, &event);
        probe_error(error, (char *)"Error enqueuing kernel\n");

        //error = clWaitForEvents(1, &event);
        //probe_error(error, (char *)"Error waiting for events\n");

        //error = clFinish(command_queue_id1);
        //probe_error(error, (char *)"Error finishing\n");

        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        //nanoSeconds = time_end - time_start;

        // execTime_upsampleDistortion_SizeId1 += nanoSeconds;

        if(TRACE_POWER)
            print_timestamp((char*) "FINISH ENQUEUE upsamplePred_SIZEID=1");    

        // execTime_upsampleDistortion += execTime_upsampleDistortion_SizeId1;



        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //          PROCEED TO CUs OF SizeID=0

        // Create kernel
        kernel_upsampleDistortion = clCreateKernel(program_sizeid0, "upsampleDistortion", &error);
        probe_error(error, (char *)"Error creating upsampleDistortion kernel\n");
        printf("Performing upsampleDistortion kernel...\n");

        // Query for work groups sizes information
        error = clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, size_ret, &preferred_size, NULL);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_WORK_GROUP_SIZE, 0, NULL, &size_ret);
        error |= clGetKernelWorkGroupInfo(kernel_upsampleDistortion, device_id, CL_KERNEL_WORK_GROUP_SIZE, size_ret, &maximum_size, NULL);

        probe_error(error, (char *)"Error querying preferred or maximum work group size\n");
        //cout << "-- Preferred WG size multiple " << preferred_size << endl;
        //cout << "-- Maximum WG size " << maximum_size << endl;

        currFrame = curr;
        // Set the arguments of the upsampleDistortion kernel
        error_1  = clSetKernelArg(kernel_upsampleDistortion, 0, sizeof(cl_mem), (void *)&return_predictionSignal_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 1, sizeof(cl_int), (void *)&frameWidth);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 2, sizeof(cl_int), (void *)&frameHeight);
        // Reference samples and final distortion
    #if MAX_PERFORMANCE_DIST // Only one distortion value is returned
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 3, sizeof(cl_mem), (void *)&return_minSadHad_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 4, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        // Unified boundariers
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 5, sizeof(cl_mem), (void *)&refT_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 6, sizeof(cl_mem), (void *)&refL_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 7, sizeof(cl_int), (void *)&currFrame);
    #else
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 3, sizeof(cl_mem), (void *)&return_SAD_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 4, sizeof(cl_mem), (void *)&return_SATD_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 5, sizeof(cl_mem), (void *)&return_minSadHad_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 6, sizeof(cl_mem), (void *)&referenceFrame_memObj);
        // Unified boundariers
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 7, sizeof(cl_mem), (void *)&refT_all_memObj);
        error_1 |= clSetKernelArg(kernel_upsampleDistortion, 8, sizeof(cl_mem), (void *)&refL_all_memObj);
    #endif

        probe_error(error_1, (char *)"Error setting arguments for the kernel\n");

        // Execute the OpenCL kernel on the list
        // These variabels are used to profile the time spend executing the kernel  "clEnqueueNDRangeKernel"
        nWG = nCTUs*NUM_CU_SIZES_SizeId0*8; // Due to the large number of CUs 4x4 per CTU (1024), each WG only processes 128 CUs (12.5%)
        global_item_size = nWG * itemsPerWG; // TODO: Correct these sizes (global and local) when considering a real scenario
        local_item_size = itemsPerWG;

        if(TRACE_POWER)
            print_timestamp((char*) "START ENQUEUE upsamplePred_SIZEID=0");    
            
        error = clEnqueueNDRangeKernel(command_queue_id0, kernel_upsampleDistortion, 1, NULL,
                                        &global_item_size, &local_item_size, 0, NULL, &event);
        probe_error(error, (char *)"Error enqueuing kernel\n");

        //error = clWaitForEvents(1, &event);
        //probe_error(error, (char *)"Error waiting for events\n");

        //error = clFinish(command_queue_id0);
        //probe_error(error, (char *)"Error finishing\n");

        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
        //clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
        //nanoSeconds = time_end - time_start;

        // execTime_upsampleDistortion_SizeId0 += nanoSeconds;
            
        if(TRACE_POWER)
            print_timestamp((char*) "FINISH ENQUEUE upsamplePred_SIZEID=0");  


        // execTime_upsampleDistortion += execTime_upsampleDistortion_SizeId0;


        // Wait until all distortions are finished to read the results into host memory
        error = clFinish(command_queue_id2);
        probe_error(error, (char *)"Error finishing\n");
        error = clFinish(command_queue_id1);
        probe_error(error, (char *)"Error finishing\n");
        error = clFinish(command_queue_id0);
        probe_error(error, (char *)"Error finishing\n");

        if(TRACE_POWER)
            print_timestamp((char*) "START READ DISTORTION");

        // READ N TIMES TO ACCOUNT FOR N KERNEL EXECUTIONS
        readMemobjsIntoArray_Distortion(command_queue_read, nCTUs, PREDICTION_MODES_ID2*2, return_SAD_memObj, return_SAD, return_SATD_memObj, return_SATD, return_minSadHad_memObj, &return_minSadHad[currFrame * nCTUs * ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES]], currFrame);
    
        // error = clFinish(command_queue_read);
        // probe_error(error, (char *)"Error finishing read\n");
        // if(TRACE_POWER){
        //     print_timestamp((char*) "FINISH READ DISTORTION");
        // }
    } // for(currFrame in N_FRAMES)

    // Wait until the distortion values are fully read
    error = clFinish(command_queue_read);
    probe_error(error, (char *)"Error finishing read\n");

    if(TRACE_POWER){
        save_finishTime();
        print_timestamp((char*) "FINISH READ DISTORTION");
    }
        

    // REPORT DISTORTION VALUES TO TERMINAL
    if(enableTerminalReport && reportDistortion){
        if(reportDistortionOnlyTarget)
            reportTargetDistortionValues_ALL(return_SAD, return_SATD, return_minSadHad, nCTUs, targetCTU);
        else
            reportAllDistortionValues_ALL(return_SAD, return_SATD, return_minSadHad, nCTUs);
    }

        

    // REPORT DISTORTION VALUES TO FILE
    if(reportDistortionToFile){
        if(reportDistortionOnlyTarget)
            reportTargetDistortionValues_File(return_SAD, return_SATD, return_minSadHad, targetCTU, frameWidth, outputFilePreffix, nCTUs);
        else
            exportAllDistortionValues_File(return_SAD, return_SATD, return_minSadHad, nCTUs, frameWidth, outputFilePreffix);
    }

    // reportTimingResults();
    reportTimingResults_Compact();


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
    error = clFlush(command_queue_common);
    error |= clFlush(command_queue_read);
    error |= clFlush(command_queue_id2);
    error |= clFlush(command_queue_id1);
    error |= clFlush(command_queue_id0);
    error |= clFinish(command_queue_common);
    error |= clFinish(command_queue_read);
    error |= clFinish(command_queue_id2);
    error |= clFinish(command_queue_id1);
    error |= clFinish(command_queue_id0);
    error |= clReleaseKernel(kernel_initRefSamples);
    error |= clReleaseKernel(kernel_reducedPrediction);
    error |= clReleaseProgram(program_sizeid2);
    error |= clReleaseProgram(program_sizeid1);
    error |= clReleaseProgram(program_sizeid0);
    error |= clReleaseCommandQueue(command_queue_common);
    error |= clReleaseMemObject(referenceFrame_memObj);
    error |= clReleaseMemObject(return_predictionSignal_memObj);
    error |= clReleaseMemObject(return_SATD_memObj);
    error |= clReleaseMemObject(return_SAD_memObj);
    error |= clReleaseMemObject(debug_mem_obj);
    error |= clReleaseMemObject(redT_all_memObj);
    error |= clReleaseMemObject(redL_all_memObj);
    error |= clReleaseMemObject(refT_all_memObj);
    error |= clReleaseMemObject(refL_all_memObj);

    probe_error(error, (char *)"Error releasing  OpenCL objects\n");

    free(source_str);
    free(platform_id);
    free(reference_frame);
    free(return_reducedPredictionSignal);
    free(return_SATD);
    free(return_SAD);
    free(debug_data);
    free(return_unified_redT); 
    free(return_unified_redL);
    free(return_unified_refT);
    free(return_unified_refL);

    if(TRACE_POWER)
        print_timestamp((char*) "FINISH HOST");

    return 0;
}