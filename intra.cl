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
        int currIdx = pass*wgSize + lid;
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
        
        /* TRACE BOUNDARIES DURING SUBSAMPLING PROCESS
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
    

    /* TRACE REDUCED BOUNDARIES
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
    
    /*
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
    //*/


    // TODO: We can put the SAD distortion computation inside the upsampling, so each workitem computes the prediction and the collocated SAD together
    // SATD is more difficult and will require a synchronizarion barrier
    upsamplePrediction_SizeId2(localPredBuffer, upsamplingHorizontal, upsamplingVertical, predictedBlock, refT, refL);    

    __local long int workgroupSad[64];
    // Each workgroup computes the SAD at a set of locations and return the total value. We must sum them to obtain the CU-level distortion
    workgroupSad[lid] = sad_64x64(predictedBlock, currentCuSamples);

    barrier(CLK_LOCAL_MEM_FENCE);


    if(lid==0){
        for(int i=1; i<wgSize; i++){
            workgroupSad[0] += workgroupSad[i];
        }
        SAD[wg] = 2*workgroupSad[0];
    }
}


// This kernel is used to fetch the reduced boundaries for all the blocks
// Each WG will process one CTU composed of a single CU size
// It works for square blocks with SizeId=2 (64x64, 32x32, 16x16)
__kernel void initReducedBoundariesSquareSizeId2(__global short *referenceFrame, const int frameWidth, const int frameHeight, __global short *redT_64x64, __global short *redL_64x64, __global short *redT_32x32, __global short *redL_32x32, __global short *redT_16x16, __global short *redL_16x16){

    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
    const short ctuIdx = wg/NUM_CU_SIZES;
    const short cuSizeIdx = wg%NUM_CU_SIZES;
    const short cuWidth = widths[cuSizeIdx];
    const short cuHeight = heights[cuSizeIdx];
    const short nCusInCtu = cusPerCtu[cuSizeIdx];
    const short itemsPerCu = wgSize/nCusInCtu;
    const short cuColumnsPerCtu = 128/cuWidth;
    const short cuRowsPerCtu = 128/cuHeight;

    // CTU position inside the frame
    const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
    const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

    __constant short reducedBoundarySize = BOUNDARY_SIZE_Id2;
    // Each of these hold one row/columns of samples for the entire CTU
    __local short int refT[128], refL[128]; 
    // These buffers are used as temporary storage between computing reduced boundaries and moving them into global memory
    __local short int bufferGlobalRedT[MAX_CUS_PER_CTU*BOUNDARY_SIZE_Id2], bufferGlobalRedL[MAX_CUS_PER_CTU*BOUNDARY_SIZE_Id2];

    // These are used to compute the reduced boundaries
    short downsamplingFactor;
    short log2DownsamplingFactor;
    short roundingOffset;

    // Index for the first sample of the CTU inside the frame
    const int idxForCtu = ctuY*frameWidth + ctuX; 

    // For MIP, the references are either direcly above or direcly left (no above-right, below-left, or top-right).
    // References are unavailable only ath the edges of the frame

    const short valueDC = 1 << 9; // Used when there are no available references
    short cuX, cuY;

    // ----------------------------------------------------
    //
    //    START COMPUTING THE REDUCED TOP BOUNDARIES
    //
    // ----------------------------------------------------

    downsamplingFactor = cuWidth/reducedBoundarySize;
    log2DownsamplingFactor = (short) log2((float) downsamplingFactor);
    roundingOffset = (1 << (log2DownsamplingFactor-1));

    // Point to the row directly above the current CTU
    int startCurrRow = idxForCtu  - frameWidth; 
   
    // Each iteration will create the redT boundary for one row of CUs inside the current CU
    for(int row=0; row<cuRowsPerCtu; row++){ 
        // Compute position of the CU this workitem is processing
        cuY = row*cuHeight;
        cuX = (lid/cuWidth)*cuWidth;
        
        // TODO: This only works if the number of workitems is equal to 128
        // Depending if there are all, some or no references available, we fetch correct samples or pad them
        if((ctuY+cuY)>0){ // Most general case, all references available
            refT[lid] = referenceFrame[startCurrRow + lid]; // At this point, one row of samples is in the shared array. We must reduce it to obtain the redT for each CU
        }
        else if((ctuY+cuY)==0 && (ctuX+cuX)==0){ // CU is in the top-left corner of frame, no reference is available. Fill with predefined DC value
            refT[lid] = valueDC;
        }
        else if((ctuY+cuY)==0 && (ctuX+cuX)>0){ // CU is in the top edge of the frame, we use the left samples to pad the top boundaries
            refT[lid] = referenceFrame[ctuX+cuX-1]; // Sample directly left of the first sample inside the CU is padded to top boundary
        }

        // Wait until all workitems have fetched the complete boundary into shared array
        barrier(CLK_LOCAL_MEM_FENCE);
        
        /*  TRACE THE COMPLETE TOP BOUNDARY FOR ONE ROW OF CUs
        if(lid==0 && wg==targetWg){
            printf("COMPLETE TOP BOUNDARY row %d\n", row);
            for(int i=0; i<128; i++){
                printf("%d,", refT[i]);
            }
            printf("\n");
        }
        //*/

        // Each workitem will compute one sample of reducedBoundary for each CU
        if(lid<(reducedBoundarySize*cuColumnsPerCtu)){ 
            // Subsample top boundary
            int temp = 0;
            for(int t=0; t<downsamplingFactor; t++){
                /*
                if(wg==targetWg && lid==1 && row==0){
                    printf("refSample,%d\n", refT[lid*downsamplingFactor + t]);
                }
                //*/
                temp += refT[lid*downsamplingFactor + t];
            }
            // Save averaged samples into local buffer. The entire buffer is moved into global memory at once in the end of processing
            bufferGlobalRedT[row*cuColumnsPerCtu*reducedBoundarySize + lid] = (temp + roundingOffset) >> log2DownsamplingFactor;
            
            /* TRACE BOUNDARIES DURING SUBSAMPLING PROCESS
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
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems computed the subsampled values
        startCurrRow += frameWidth*cuHeight; // Update the beginning of the next row of samples (i.e., top of next row of CUs)
    }

    /* TRACE REDUCED TOP BOUNDARIES FOR THE ENTIRE CTU
    if(wg==targetWg && lid==0){
        printf("All reduced ref T\n");
        for(int i=0; i<8*8*4; i++){
            printf("%d,", bufferGlobalRedT[i]);
        }
        printf("\n\n");
    }
    //*/

    // --------    MOVE REDUCED TOP BOUNDARY FROM LOCAL BUFFER INTO GLOBAL MEMORY
    // TODO: Review these if/elses if the workgroup size is different from 128 (it may required more or less passes in smaller CU sizes) 
    if(cuSizeIdx == _64x64){
        if(lid < 4*2*2){
            redT_64x64[ctuIdx*nCusInCtu*reducedBoundarySize + lid] = bufferGlobalRedT[lid];        
        }
    }
    else if(cuSizeIdx == _32x32){
        if(lid < 4*4*4){
            redT_32x32[ctuIdx*nCusInCtu*reducedBoundarySize + lid] = bufferGlobalRedT[lid];
        }
    }
    else if(cuSizeIdx == _16x16){
        if(lid < 4*8*8){
            int nPasses = (4*8*8)/wgSize; // nPasses=2 for wgSize=128 and CUs 16x16 aligned
            for(int pass=0; pass<nPasses; pass++){
                redT_16x16[ctuIdx*nCusInCtu*reducedBoundarySize + pass*wgSize + lid] = bufferGlobalRedT[pass*wgSize + lid];
            }
        }
    }

    // --------------------------------------------------
    //     AT THIS POINT ALL REDUCED TOP SAMPLES ARE IN GLOBAL MEMORY
    //     WE CAN ADD A SYNCH BARRIER AND REUSE THE bufferGlobalRedT
    //     FOR THE ELFT BOUNDARY OR SIMPLY IGNORE IT, AVOID THE SYNCH 
    //     BARRIER AND CREATE ANOTHER BUFFER FOR LEFT BOUNDARY
    // --------------------------------------------------


    // ----------------------------------------------------
    //
    //    NOW WE COMPUTE THE REDUCED LEFT BOUNDARIES
    //
    // ----------------------------------------------------

    downsamplingFactor = cuHeight/reducedBoundarySize;
    log2DownsamplingFactor = (short) log2((float) downsamplingFactor);
    roundingOffset = (1 << (log2DownsamplingFactor-1));

    // Point to the column directly left of the current CTU
    int startCurrCol = idxForCtu  - 1; 
    
    // Each iteration will create the redL boundary for one column of CUs inside the current CU
    // Since the CU data is stored in raster order and we are processing CUs in different lines (i.e., strided CUs), we must store
    // The results in strided manner in local buffer
    for(int col=0; col<cuColumnsPerCtu; col++){ 
        cuY = (lid/cuHeight)*cuHeight;
        cuX = col*cuWidth;
        // TODO: This only works if the number of workitems is equal to 128
        // TODO: This is VERY INEFFICIENT since the global memory accesses are completely strided
        // TODO: If we store the referenceFrame in original and transposed manners (i.e., keep the same data in two different global memory objects, one transposed and other not)
        //       It is possible to coalesce the memory accesses to refL as well. TOP boundaries accessed from ORIGINAL FRAME and LEFT boundaries accessed from TRANSPOSED FRAME
        if((ctuX+cuX)>0){ // Most general case, all neighboring samples are available
            refL[lid] = referenceFrame[startCurrCol + lid*frameWidth]; // At this point, one row of samples is in the shared array. We must reduce it to obtain the redL for each CU
        }
        else if((ctuY+cuY)==0 && (ctuX+cuX)==0){ // CU is in the top-left corner of frame, no reference is available. Fill with predefined DC value
            refL[lid] = valueDC;
        }
        else if((ctuX+cuX)==0 && (ctuY+cuY)>0){ // CU is in the left edge of the frame, we use the top samples to pad the left boundaries
            refL[lid] = referenceFrame[(ctuY+cuY-1)*frameWidth];  // Sample directly above of the first sample inside the CU is padded to left boundary
        }

        // Wait until all workitems have fetched the complete boundary into shared array
        barrier(CLK_LOCAL_MEM_FENCE);
        /*   TRACE THE COMPLETE LEFT BOUNDARY FOR ONE COLUMNS OF CUs
        if(lid==0 && wg==targetWg){
            printf("COMPLETE LEFT BOUNDARY col %d\n", col);
            for(int i=0; i<128; i++){
                printf("%d,", refL[i]);
            }
            printf("\n");
        }
        //*/

        // Each workitem will compute one sample of reducedBoundary for each CU
        if(lid<(reducedBoundarySize*cuRowsPerCtu)){ 
            // Subsample top boundary
            int temp = 0;
            int currentRow = lid/reducedBoundarySize;
            for(int t=0; t<downsamplingFactor; t++){
                /* 
                if(wg==targetWg && lid==0 && col==0){
                    printf("refSample,%d\n", refL[lid*downsamplingFactor + t]);
                }
                //*/
                temp += refL[lid*downsamplingFactor + t];
            }

            // Although we are computing the reduced boundaries of an entire column at once, the boundaries are stored in shared memory using a CU-raster order
            // First the boundaries of the top-left CU are store in memory, then the boundaries of the CU to the right and so on. The accesses to shared memory are very strided
            // The entire buffer is moved into global memory at once in the end of processing
            bufferGlobalRedL[currentRow*cuColumnsPerCtu*reducedBoundarySize + col*reducedBoundarySize + lid%reducedBoundarySize] = (temp + roundingOffset) >> log2DownsamplingFactor;
            
            /* TRACE BOUNDARIES DURING SUBSAMPLING PROCESS
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
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems computed the subsampled values
        startCurrCol += cuWidth; // Update the beginning of the next column  of samples (i.e., left of next column of samples)
    }

    /* TRACE REDUCED TOP BOUNDARIES FOR THE ENTIRE CTU
    if(wg==targetWg && lid==0){
        printf("All reduced ref L\n");
        for(int i=0; i<4*cuRowsPerCtu*cuColumnsPerCtu; i++){
            printf("%d,", bufferGlobalRedL[i]);
        }
        printf("\n\n");
    }
    //*/

    // --------    MOVE REDUCED LEFT BOUNDARY FROM LOCAL BUFFER INTO GLOBAL MEMORY
    // TODO: Review these if/elses if the workgroup size is different from 128 (it may required more or less passes in smaller CU sizes) 
    if(cuSizeIdx == _64x64){
        if(lid < 4*2*2){
            redL_64x64[ctuIdx*nCusInCtu*reducedBoundarySize + lid] = bufferGlobalRedL[lid];        
        }
    }
    else if(cuSizeIdx == _32x32){
        if(lid < 4*4*4){
            redL_32x32[ctuIdx*nCusInCtu*reducedBoundarySize + lid] = bufferGlobalRedL[lid];
        }
    }
    else if(cuSizeIdx == _16x16){
        if(lid < 4*8*8){
            int nPasses = (4*8*8)/wgSize;
            for(int pass=0; pass<nPasses; pass++){ // nPasses=2 for wgSize=128 and CUs 16x16 aligned
                redL_16x16[ctuIdx*nCusInCtu*reducedBoundarySize + pass*wgSize + lid] = bufferGlobalRedL[pass*wgSize + lid];
            }
        }
    }
}


// This kernel is used to obtain the reduced prediction with all prediction modes
// The prediction of all prediction modes is stored in global memory and returned to the host
// Each WG will process one CTU composed of a single CU size
// It works for square blocks with SizeId=2 (64x64, 32x32, 16x16)
__kernel void MIP_squareSizeId2(__global short *redT_64x64, __global short *redL_64x64, __global short *redT_32x32, __global short *redL_32x32, __global short *redT_16x16, __global short *redL_16x16, __global short *reducedPrediction, const int frameWidth, const int frameHeight){
    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    const short ctuIdx = wg/NUM_CU_SIZES;
    const short cuSizeIdx = wg%NUM_CU_SIZES;
    const short cuWidth = widths[cuSizeIdx];
    const short cuHeight = heights[cuSizeIdx];
    const short nCusInCtu = cusPerCtu[cuSizeIdx];

    const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
      
    const short cuColumnsPerCtu = 128/cuWidth;
    const short cuRowsPerCtu = 128/cuHeight;

    // CTU position inside the frame
    const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
    const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

    int boundaryStrideForCtu = cusPerCtu[cuSizeIdx]*BOUNDARY_SIZE_Id2; // Each CU occupy BOUNDARY_SIZE_Id2 (=4) positions in the reduced boundaries buffers
    int currCtuBoundariesIdx = ctuIdx * boundaryStrideForCtu;

    // This buffer stores all predicted CUs inside the current CTU, with a single prediction mode
    // Each CU is processed by 64 workitems, where each workitem conducts the prediction of a single sample
    // When necessary, each workitem will process more than one CU
    // After all CUs are predicted with a single prediction mode, the buffer is moved into global memory and the next prediction mode is tested
    __local short reducedPredictedCtu[MAX_CUS_PER_CTU][(BOUNDARY_SIZE_Id2*2)*(BOUNDARY_SIZE_Id2*2)];

    int totalPredictionModes = PREDICTION_MODES_ID2 + TEST_TRANSPOSED_MODES*PREDICTION_MODES_ID2;
    // Each 64 workitems process one CU irrespective of the "original size" since the reduced prediciton is 64x64 for all of them
    const short itemsPerCu = 64;
    const char sampleInCu = lid%itemsPerCu; // Each workitem processes 1 sample
    int cusPerPass = wgSize/itemsPerCu;
    int nPasses = nCusInCtu / cusPerPass;
    short cuIdxInCtu;
    int predSample;
    
    short8 reducedBoundaries, coefficients;
    short4 reducedT, reducedL;

    // TODO: Change this for loop to fetch the boundaries a single time for non-transposed and a single time for transposed
    // Use for(transp = [0,1]){ for(mode = [0,5])}
    for(int m=0; m<totalPredictionModes; m++){
        
        short mode = m%PREDICTION_MODES_ID2;
        short t = -(m/PREDICTION_MODES_ID2); // -1 because this value is used in select(), and select() only tests the MSB of the value
        short8 isTransp = (short8) (t,t,t,t,t,t,t,t);

        for(int pass=0; pass<nPasses; pass++){
            cuIdxInCtu = pass*cusPerPass + floor((float)lid/itemsPerCu);
            // Fetch the redT and redL samples from global memory
            if(cuWidth == 64){
                reducedT = vload4((currCtuBoundariesIdx + cuIdxInCtu*BOUNDARY_SIZE_Id2)/4, redT_64x64);
                reducedL = vload4((currCtuBoundariesIdx + cuIdxInCtu*BOUNDARY_SIZE_Id2)/4, redL_64x64);
            }
            else if(cuWidth == 32){
                reducedT = vload4((currCtuBoundariesIdx + cuIdxInCtu*BOUNDARY_SIZE_Id2)/4, redT_32x32);
                reducedL = vload4((currCtuBoundariesIdx + cuIdxInCtu*BOUNDARY_SIZE_Id2)/4, redL_32x32);
            }
            else if(cuWidth == 16){
                reducedT = vload4((currCtuBoundariesIdx + cuIdxInCtu*BOUNDARY_SIZE_Id2)/4, redT_16x16);
                reducedL = vload4((currCtuBoundariesIdx + cuIdxInCtu*BOUNDARY_SIZE_Id2)/4, redL_16x16);
            }

            // Create complete boundaries array based on transposed or not-transposed
            reducedBoundaries = select((short8)(reducedT, reducedL), (short8)(reducedL, reducedT), isTransp);

            short firstVal = reducedBoundaries.s0;
            // Apply inputOffset to all boundaries except the first, then zero the first. After this the boundaries are ready to be multiplied by the coefficients
            reducedBoundaries = reducedBoundaries - (short8) (firstVal); //(0, firstVal, firstVal, firstVal, firstVal, firstVal, firstVal, firstVal);

            int offset = reducedBoundaries.s0 + reducedBoundaries.s1 + reducedBoundaries.s2 + reducedBoundaries.s3 + reducedBoundaries.s4 + reducedBoundaries.s5 + reducedBoundaries.s6 + reducedBoundaries.s7;
            offset = (1 << (MIP_SHIFT_MATRIX - 1)) - MIP_OFFSET_MATRIX * offset;

            coefficients.s0 = 0;
            coefficients.s1 = mipMatrix16x16[mode][sampleInCu][0];
            coefficients.s2 = mipMatrix16x16[mode][sampleInCu][1];
            coefficients.s3 = mipMatrix16x16[mode][sampleInCu][2];
            coefficients.s4 = mipMatrix16x16[mode][sampleInCu][3];
            coefficients.s5 = mipMatrix16x16[mode][sampleInCu][4];
            coefficients.s6 = mipMatrix16x16[mode][sampleInCu][5];
            coefficients.s7 = mipMatrix16x16[mode][sampleInCu][6];

            // TODO: Use some dot-product or MAC optimization to conduct this multiplication-add
            predSample =  offset;
            predSample += coefficients.s1 * reducedBoundaries.s1;
            predSample += coefficients.s2 * reducedBoundaries.s2;
            predSample += coefficients.s3 * reducedBoundaries.s3;
            predSample += coefficients.s4 * reducedBoundaries.s4;
            predSample += coefficients.s5 * reducedBoundaries.s5;
            predSample += coefficients.s6 * reducedBoundaries.s6;
            predSample += coefficients.s7 * reducedBoundaries.s7;

            predSample = (predSample >> MIP_SHIFT_MATRIX) + firstVal;
            predSample = clamp(predSample, 0, (1<<10)-1);

            reducedPredictedCtu[cuIdxInCtu][sampleInCu] = predSample;

            // Wait until all samples of the CTU are predicted because we will move it into global memory in sequence
            barrier(CLK_LOCAL_MEM_FENCE);

            /*    TRACE PREDICTION
            if(1 && lid==(192+32) && cuWidth==64){
                for(int cu=0; cu<4; cu++){
                    printf("REDUCED PREDICTION: CTU %d, CU %d, Mode %d\n", ctuIdx, cu, m);
                    for(int i=0; i<8; i++){
                        for(int j=0; j<8; j++){
                            printf("%d,", reducedPredictedCtu[cu][i*8+j]);
                        }
                        printf("\n");
                    }
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/
        }
        // Now we move the prediction from __local into __global memory
        // This points to the start of current CTU in the reduced prediction global buffer
        const int currCtuPredictionIdx = ctuIdx*TOTAL_CUS_PER_CTU*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
        int currCuPredictionIdx; // This will point to the start of the current CU in the reduced prediction global buffer (i.e., CTU position + offset)
        for(int pass=0; pass<nPasses; pass++){
            cuIdxInCtu = pass*cusPerPass + floor((float)lid/itemsPerCu);
            // Point to start of this CU size in global buffer
            currCuPredictionIdx = currCtuPredictionIdx + stridedCusPerCtu[cuSizeIdx]*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
            // Point to start of this CU specifically in global buffer
            currCuPredictionIdx += cuIdxInCtu*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
            // Point to start of the current mode in global buffer
            currCuPredictionIdx += m*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2;

            // When the prediction uses transposed modes, the reduced prediction signal must be transposed when moved into global memory
            // This is used to translate a non-transposed index into a transposed index, and access global memory more efficiently
            int sampleInTransposedCU; 

            int ntX, ntY, tX, tY;
            ntX = sampleInCu%REDUCED_PRED_SIZE_Id2;
            ntY = sampleInCu/REDUCED_PRED_SIZE_Id2;
            tX = ntY; // Swap non-transposed and transposed X/Y coordinates
            tY = ntX;
            // Transform the coordinates into a one-dimensional array for the 8*8=64 reduced prediction block
            sampleInTransposedCU = tY*REDUCED_PRED_SIZE_Id2 + tX;
            // In transposed modes, t=-1. This selects between transposed and non-transposed indices with a simple operation
            int correctedSampleInCu = -1*t*sampleInTransposedCU + (1+t)*sampleInCu;

            reducedPrediction[currCuPredictionIdx + sampleInCu] = reducedPredictedCtu[cuIdxInCtu][correctedSampleInCu];
            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all predicted samples are moved. The next iteration overwrites the local prediction buffer
        }
    } // End of current mode
}