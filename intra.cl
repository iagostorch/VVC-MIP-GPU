#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#pragma OPENCL EXTENSION cl_nv_compiler_options : enable

#endif


__kernel void MIP_64x64(__global short *currBlock_memObj, __global short *refSamples_memObj,const int TOTAL_PREDICTION_MODES,  __global short *predictedBlock, __global long *SATD, __global long *SAD){
    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    predictedBlock[lid] = gid;

}
