#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#pragma OPENCL EXTENSION cl_nv_compiler_options : enable

#endif

#include "mip_matrix.cl"
#include "kernel_aux_functions.cl"

// The prediction is computed in void MatrixIntraPrediction::predBlock
__kernel void MIP_64x64(__global short *currBlock_memObj, __global short *refSamples_memObj,const int TOTAL_PREDICTION_MODES,  __global short *predictedBlock, __global long *SATD, __global long *SAD, __global short *debug){
    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    __constant int BLOCK_WIDTH = 64;
    __constant int BLOCK_HEIGHT = 64;

    int width = BLOCK_WIDTH; //widths[lid%17];
    int height = BLOCK_HEIGHT; //heights[lid%17];

    int isSize0 = (width==4) && (height==4);
    int isSize1 = (width==8 && height==8) || (!isSize0 && (width==4 || height==4));
    int isSize2 = !isSize1 && (width>=8 || height>=8);

    // TODO: Correct when supporting more block sizes
    // The general implementation is below this line
    int sizeId = 2;
    // int sizeId = select(-1, 0, isSize0);
    // sizeId = select(sizeId, 1, isSize1);
    // sizeId = select(sizeId, 2, isSize2);

    //TODO: Correct this when supporting more block sizes
    // Correct value is select(8, 4, sizeId<2);
    int reducedPredSize = REDUCED_PRED_SIZE_Id2;


    // TODO: Correct when supporting more block sizes
    __local short currentCuSamples[64*64];
    int nPasses = (64*64)/wgSize;
    for(int pass=0; pass<nPasses; pass++){
        // TODO: Correct ti support CUs in different positions. For now, the kernel input is a single CU and its references
        int currIdx = pass*64*64 + lid;
        currentCuSamples[currIdx] = currBlock_memObj[currIdx];
    }

    // In MIP prediction, only the directly above and directly left samples are references. We can ignore the below-left, above-right, and top-left
    __local short refL[64], refT[64];
    __constant short startTop = 1; // Skip the top-left reference
    __constant short startLeft = 1 + 2*64 +1; // Skip the top-left, above, and above right. The skip the top-left again (it is placed twice in the list)

    refT[lid] = refSamples_memObj[startTop + lid];
    refL[lid] = refSamples_memObj[startLeft + lid];

    barrier(CLK_LOCAL_MEM_FENCE);
    
    /* TRACE BOUNDARIES
    if(gid==0){
        printf("TOP COMPLETE BOUNDARY\n");
        for(int i=0; i<64; i++){
            printf("%d,", refT[i]);
        }
        printf("\n\n\n");

        printf("LEFT COMPLETE BOUNDARY\n");
        for(int i=0; i<64; i++){
            printf("%d,", refL[i]);
        }
        printf("\n\n\n");
    }
    //*/
    __constant short reducedBoundarysize = BOUNDARY_SIZE_Id2;
    __local short redL[BOUNDARY_SIZE_Id2], redT[BOUNDARY_SIZE_Id2], redLT[2*BOUNDARY_SIZE_Id2], redTL[2*BOUNDARY_SIZE_Id2]; // the last two buffers refLT and redTL hold the two types of references simultaneously, in original and transposed manner. They are equivalent to pTemp[]

    /*
    ########################################################################

    SUMSAMPLE THE REFERENCE SAMPLES PRIOR T MIP PREDICTION

    This snippet is based on MatrixIntraPrediction::boundaryDownsampling1D
    
    ########################################################################
    */
    short downsamplingFactor, log2DownsamplingFactor, roundingOffset;

    // Downsampling of top references
    downsamplingFactor = width/reducedBoundarysize;
    log2DownsamplingFactor = (short) log2((float) downsamplingFactor);
    roundingOffset = (1 << (log2DownsamplingFactor-1));

    // TODO: We could use one kernel just to generate the references, and another one to conduct the prediction
    // This way it is possible to optimize the number of workitems per workgroup
    if(lid<reducedBoundarysize){
        // Subsample top boundary
        int temp = 0;
        for(int t=0; t<downsamplingFactor; t++){
            temp += refT[lid*downsamplingFactor + t];
        }
        
        redT[lid] = (temp + roundingOffset) >> log2DownsamplingFactor;
        
        //* TRACE BOUNDARIES DURING SUBSAMPLING PROCESS
        if(wg==targetWg){
            printf("Summed values top [%d] = %d\n", lid, temp);
            printf("Processed values top [%d] = %d\n", lid, (temp + roundingOffset) >> log2DownsamplingFactor);
        }
        //*
        
        // Subsample left boundary
        temp = 0;
        for(int t=0; t<downsamplingFactor; t++)
            temp += refL[lid*downsamplingFactor + t];

        redL[lid] = (temp + roundingOffset) >> log2DownsamplingFactor;
        
        //* TRACE BOUNDARIES DURING SUBSAMPLING PROCESS
        if(wg==targetWg){
            printf("Summed values left [%d] = %d\n", lid, temp);
            printf("Processed values left [%d] = %d\n", lid, (temp + roundingOffset) >> log2DownsamplingFactor);
        }
        //*/

        redTL[lid] = redT[lid];
        redTL[lid+reducedBoundarysize] = redL[lid];
        redLT[lid+reducedBoundarysize] = redT[lid];
        redLT[lid] = redL[lid];       
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int hasFirstCol = sizeId<2; // Always false for 64x64
    short inputOffset = redTL[0];
    short inputOffsetTransp = redLT[0];

    // TODO: Verify the need to use these two barriers in sequence
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i=1; i<2*reducedBoundarysize; i++){
        redTL[i] = redTL[i] - inputOffset;
        redLT[i] = redLT[i] - inputOffsetTransp;
    }

    // TODO: I believe this is an implementation optimization from VTM cause the paper does not mention this exception
    // Since the first coefficient for sizeID=2 is always zero, it makes no difference what the first reference is (therefor, VTM forces zero)
    // Lets remove this exception to avoid if/else and divergent branching
    // if(lid==0){
    //     redTL[0] = 0; //(1<<(10-1))-inputOffset;
    //     redLT[0] = 0; //(1<<(10-1))-inputOffsetTransp;
    // }
    

    //* TRACE REDUCED BOUNDARIES
    if(wg==targetWg && lid==targetLid){
        printf("Reduced TL\n");
        for(int i=0; i<8; i++){
            printf("%d,", redTL[i]);
        }
        printf("\n");

        printf("Reduced LT\n");
        for(int i=0; i<8; i++){
            printf("%d,", redLT[i]);
        }
        printf("\n");
    }
    //*/

    /*
    ########################################################################

    AT THIS POINT THE REFERENCE SAMPLES ARE ALREADY SUBSAMPLED IN redLT and redTL buffers
    
    ########################################################################
     */

    /*
    ########################################################################

    PREDICT THE CURRENT BLOCK ACCORDING TO ONE PREDICTION MODE

    This snippet is based on IntraPrediction::predIntraMip
    
    ########################################################################
     */

    int upsamplingHorizontal = width/reducedPredSize;
    int upsamplingVertical = height/reducedPredSize;
    
    __local short localPredBuffer[REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2];

    // TODO: Correct this when supporting more block sizes.
    // Correct value is upsamplingHorizontal>1 || upsamplingVertical>1;
    int needUpsampling = 1; 
    
    int mode = wg%6;
    short isTransposed = wg>=6;

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int offset = 0;
    for(int i=1; i<reducedPredSize; i++){
        offset += select(redTL[i], redLT[i], isTransposed); 
    }
    offset = (1 << (MIP_SHIFT_MATRIX - 1)) - MIP_OFFSET_MATRIX * offset;
    
    // TODO: This only works for sizeId = 2. Correct when supporting mroe block sizes. This is used to discard the first column of pred matrix
    const int redSize = 1;

    // At this point, each workitem is going to predict a single sample in the reducedPred, at position (x,y)
    int predSample = offset; // Offset is always added to the predicted sample
    short xPos = lid%REDUCED_PRED_SIZE_Id2;
    short yPos = lid/REDUCED_PRED_SIZE_Id2;

    // TODO: for sizeId=2, the first column of the matrix is always zero so the coefficients actually start at 1
    // The first share of predSample is always 0
    predSample += 0;
    // NOTE the offset between redTL[i] and mipMatrix16x16[][][i-1] to comply with the first columns being all zeros
    // TODO: This multiplication can be improved using the dot product, if the two arrays are stored as int8
    predSample += select(redTL[1], redLT[1], isTransposed)*mipMatrix16x16[mode][yPos*REDUCED_PRED_SIZE_Id2+xPos][0];
    predSample += select(redTL[2], redLT[2], isTransposed)*mipMatrix16x16[mode][yPos*REDUCED_PRED_SIZE_Id2+xPos][1];
    predSample += select(redTL[3], redLT[3], isTransposed)*mipMatrix16x16[mode][yPos*REDUCED_PRED_SIZE_Id2+xPos][2];
    predSample += select(redTL[4], redLT[4], isTransposed)*mipMatrix16x16[mode][yPos*REDUCED_PRED_SIZE_Id2+xPos][3];
    predSample += select(redTL[5], redLT[5], isTransposed)*mipMatrix16x16[mode][yPos*REDUCED_PRED_SIZE_Id2+xPos][4];
    predSample += select(redTL[6], redLT[6], isTransposed)*mipMatrix16x16[mode][yPos*REDUCED_PRED_SIZE_Id2+xPos][5];
    predSample += select(redTL[7], redLT[7], isTransposed)*mipMatrix16x16[mode][yPos*REDUCED_PRED_SIZE_Id2+xPos][6];
    
    predSample = (predSample >> MIP_SHIFT_MATRIX) + select(inputOffset, inputOffsetTransp, isTransposed);
    predSample = clamp(predSample, 0, (1<<10)-1);

    // If the current mode is transposed, the workitems access the local buffer in a strided manner and the global memory in a coalesced manner
    // If it is not transposed, both global and local memory are accessed in a coalesced manner
    short xPosTranspCompliant = select(xPos, yPos, isTransposed);
    short yPosTranspCompliant = select(yPos, xPos, isTransposed);

    localPredBuffer[yPosTranspCompliant*REDUCED_PRED_SIZE_Id2 + xPosTranspCompliant] = predSample;

    // In case the current mode is transposed, it is necessary to copy the predicted samples from local to global memory in a non-efficient fashion
    // By synchronizing all workitems, we can change what items access what samples to make them access global memory in a more efficient manner
    barrier(CLK_LOCAL_MEM_FENCE);
    int globalStride = wg*64*64;
    
    if(wg==targetWg && lid==targetLid){
        printf("Reduced prediction\n");
        for(int i=0; i<8; i++){
            for(int j=0; j<8; j++){
                printf("%d,", localPredBuffer[i*8+j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    upsamplePrediction_SizeId2(localPredBuffer, upsamplingHorizontal, upsamplingVertical, predictedBlock, refT, refL);    
}
