#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#pragma OPENCL EXTENSION cl_nv_compiler_options : enable

#endif

#include "mip_matrix.cl"
#include "kernel_aux_functions.cl"

// This kernel is used to fetch the reduced boundaries for all the blocks
// Each WG will process one CTU composed of a single CU size
// It works for all blocks with SizeId=2 and all alignments
__kernel void initBoundariesSquareSizeId2_ALL(__global short *referenceFrame, const int frameWidth, const int frameHeight, __global short *unified_redT, __global short *unified_redL, __global short *unified_refT, __global short *unified_refL){

    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
    const short ctuIdx = wg/ALL_NUM_CU_SIZES;
    const short cuSizeIdx = wg%ALL_NUM_CU_SIZES;
    const short cuWidth = ALL_widths[cuSizeIdx];
    const short cuHeight = ALL_heights[cuSizeIdx];
    const short nCusInCtu = ALL_cusPerCtu[cuSizeIdx];
    const short itemsPerCu = wgSize/nCusInCtu;
    const short cuColumnsPerCtu = ALL_cuColumnsPerCtu[cuSizeIdx];
    const short cuRowsPerCtu = ALL_cuRowsPerCtu[cuSizeIdx];

    // CTU position inside the frame
    const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
    const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

    char reducedBoundarySize = ALL_reducedBoundarySizes[cuSizeIdx];
    // Each of these hold one row/columns of samples for the entire CTU
    __local short int refT[128], refL[128]; 
    // These buffers are used as temporary storage between computing reduced boundaries and moving them into global memory
    __local short int bufferGlobalRedT[2048], bufferGlobalRedL[2048]; // Maximum value. 1024 CUs 4x4 with reducedBoundary=2 each

    __local short int bufferGlobalRefT[MAX_CU_ROWS_PER_CTU][128], bufferGlobalRefL[MAX_CU_COLUMNS_PER_CTU][128];

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
    short cuY_4x4, cuX_4x4; // Used to compute a temporary coordinate for 4x4 CUs to avoid using tons of const memory

    // ----------------------------------------------------
    //
    //    START COMPUTING THE REDUCED TOP BOUNDARIES
    //
    // ----------------------------------------------------

    downsamplingFactor = cuWidth/reducedBoundarySize;
    log2DownsamplingFactor = (short) log2((float) downsamplingFactor);
    roundingOffset = (1 << (log2DownsamplingFactor-1));

    // Point to the row directly above the current CTU
    int anchorCurrCtu = idxForCtu  - frameWidth; 
    int startCurrRow;
    
    // Each iteration will create the redT boundary for one or more rows of CUs inside the current CU
    int rowsPerPass = 128/(cuColumnsPerCtu*cuWidth); // Either 1, 2, 4
    for(int firstRow=0; firstRow<cuRowsPerCtu; firstRow+=rowsPerPass){ 
        int row = firstRow + lid/(cuColumnsPerCtu*cuWidth); // Row that this workitem is processing
        int cuIdx = firstRow*cuColumnsPerCtu + lid/cuWidth;

        // Compute position of the CU this workitem is processing
        cuY_4x4 = 4*(cuIdx/cuColumnsPerCtu);
        cuX_4x4 = 4*(cuIdx%cuColumnsPerCtu);

        cuY = select((short) ALL_Y_POS[cuSizeIdx][cuIdx], cuY_4x4, (short) (cuSizeIdx==ALL_AL_4x4)); 
        cuX = select((short) ALL_X_POS[cuSizeIdx][cuIdx], cuX_4x4, (short) (cuSizeIdx==ALL_AL_4x4)); 

        startCurrRow = anchorCurrCtu + cuY*frameWidth;
 
        // TODO: This only works if the number of workitems is equal to 128
        // Depending if there are all, some or no references available, we fetch correct samples or pad them
        if((ctuY+cuY)>0){ // Most general case, all references available
            refT[lid] = referenceFrame[startCurrRow + cuX + lid%cuWidth]; // At this point, one row of samples is in the shared array. We must reduce it to obtain the redT for each CU
        }
        else if((ctuY+cuY)==0 && (ctuX+cuX)==0){ // CU is in the top-left corner of frame, no reference is available. Fill with predefined DC value
            refT[lid] = valueDC;
        }
        else if((ctuY+cuY)==0 && (ctuX+cuX)>0){ // CU is in the top edge of the frame, we use the left samples to pad the top boundaries
            refT[lid] = referenceFrame[ctuX+cuX-1]; // Sample directly left of the first sample inside the CU is padded to top boundary
        }
        bufferGlobalRefT[row][lid%(cuColumnsPerCtu*cuWidth)] = refT[lid];
        // Wait until all workitems have fetched the complete boundary into shared array
        barrier(CLK_LOCAL_MEM_FENCE);
        
        /*  TRACE THE COMPLETE TOP BOUNDARY FOR ONE ROW OF CUs
        //if(lid==0 && wg==targetWg){
        if(1 && cuSizeIdx==ALL_NA_8x16_G4 && ctuIdx==16 && lid==0){
            for(int r=firstRow; r<firstRow+rowsPerPass; r++){
                printf("COMPLETE TOP BOUNDARY row %d\n", r);
                for(int i=0; i<cuColumnsPerCtu*cuWidth; i++){
                    //printf("%d,", refT[i]);
                    printf("%d,", bufferGlobalRefT[r][i]); 
                }
                printf("\n");
            }
        }
        //*/

        // Each workitem will compute one sample of reducedBoundary for each CU
        if(lid<(reducedBoundarySize*cuColumnsPerCtu*rowsPerPass)){ 
            row = firstRow + lid/(cuColumnsPerCtu*reducedBoundarySize); // Row that this workitem is processing
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
            bufferGlobalRedT[row*cuColumnsPerCtu*reducedBoundarySize + lid%(cuColumnsPerCtu*reducedBoundarySize)] = (temp + roundingOffset) >> log2DownsamplingFactor;
            
            /* TRACE BOUNDARIES DURING SUBSAMPLING PROCESS
            if(wg==targetWg){
                printf("Summed values top [%d] = %d\n", lid, temp);
                printf("Processed values top [%d] = %d\n", lid, (temp + roundingOffset) >> log2DownsamplingFactor);
            }
            //*/
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems computed the subsampled values
    }

    /* TRACE REDUCED TOP BOUNDARIES FOR THE ENTIRE CTU
    if(wg==targetWg && lid==0){
        printf("All reduced ref T: CTU=%d, cuSizeIdx=%d\n", ctuIdx, cuSizeIdx);
        for(int i=0; i<ALL_cusPerCtu[cuSizeIdx]*reducedBoundarySize; i++){
            printf("%d,", bufferGlobalRedT[i]);
        }
        printf("\n\n");
    }
    //*/

    // ---------   UNIFIED BUFFERS FOR TOP BOUNDARIES
    //
    // Move reduced boundary into unified global memory buffer
    if(lid < reducedBoundarySize*nCusInCtu){
        int idx;
        int nPasses = max(1, (reducedBoundarySize*nCusInCtu)/wgSize); // 
        for(int pass=0; pass<nPasses; pass++){
            // Point to current CTU
            idx = ctuIdx*(ALL_TOTAL_CUS_SizeId12_PER_CTU*BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU*BOUNDARY_SIZE_Id0);
            // Point to current CU size. Even though CUs 4x4 have reducedBoundarySize=2, the ALL_stridedCusPerCtu points to the start of the current CU size and all previous sizes have reducedboundarySize=4
            idx += ALL_stridedCusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY;
            
            unified_redT[idx + pass*wgSize + lid] = bufferGlobalRedT[pass*wgSize + lid];
        }
    }
    
    // Move complete boundary into global memory
    if(lid < (cuRowsPerCtu*cuColumnsPerCtu*cuWidth)){
        
        int nRows = cuRowsPerCtu;
        int rowsPerPass = wgSize/(cuColumnsPerCtu*cuWidth);
        int nPasses = max(1, (cuRowsPerCtu*cuColumnsPerCtu*cuWidth) / wgSize);
        for(int pass=0; pass<nPasses; pass++){
            int currRow = pass*rowsPerPass + lid/(cuColumnsPerCtu*cuWidth);
            int cuInRow = (lid/cuWidth)%cuColumnsPerCtu;
            unified_refT[ctuIdx*ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES] + ALL_stridedCompleteTopBoundaries[cuSizeIdx] + (currRow*cuColumnsPerCtu+cuInRow)*cuWidth + lid%cuWidth] = bufferGlobalRefT[currRow][cuInRow*cuWidth + lid%cuWidth];
        }
    }

    // ----------------------------------------------------
    //
    //    NOW WE COMPUTE THE REDUCED LEFT BOUNDARIES
    //
    // ----------------------------------------------------

    downsamplingFactor = cuHeight/reducedBoundarySize;
    log2DownsamplingFactor = (short) log2((float) downsamplingFactor);
    roundingOffset = (1 << (log2DownsamplingFactor-1));

    // Point to the column directly left of the current CTU
    anchorCurrCtu = idxForCtu  - 1; 
    int startCurrCol;
    
    // Each iteration will create the redL boundary for one column of CUs inside the current CU
    // Since the CU data is stored in raster order and we are processing CUs in different lines (i.e., strided CUs), we must store
    // The results in strided manner in local buffer
    int colsPerPass = 128/(cuRowsPerCtu*cuHeight);
    for(int firstCol=0; firstCol<cuColumnsPerCtu; firstCol+=colsPerPass){ 
        int col = firstCol + lid/(cuRowsPerCtu*cuHeight);
        int row = (lid/cuHeight)%cuRowsPerCtu;
        int cuIdx = row*cuColumnsPerCtu + col;

        // Compute position of the CU this workitem is processing
        cuY_4x4 = 4*(cuIdx/cuColumnsPerCtu);
        cuX_4x4 = 4*(cuIdx%cuColumnsPerCtu);
        
        cuY = select((short) ALL_Y_POS[cuSizeIdx][cuIdx], cuY_4x4, (short) (cuSizeIdx==ALL_AL_4x4)); 
        cuX = select((short) ALL_X_POS[cuSizeIdx][cuIdx], cuX_4x4, (short) (cuSizeIdx==ALL_AL_4x4));

        int startCurrCol = anchorCurrCtu + cuX;

        // TODO: This only works if the number of workitems is equal to 128
        // TODO: This is VERY INEFFICIENT since the global memory accesses are completely strided
        // TODO: If we store the referenceFrame in original and transposed manners (i.e., keep the same data in two different global memory objects, one transposed and other not)
        //       It is possible to coalesce the memory accesses to refL as well. TOP boundaries accessed from ORIGINAL FRAME and LEFT boundaries accessed from TRANSPOSED FRAME
        if((ctuX+cuX)>0){ // Most general case, all neighboring samples are available
            refL[lid] = referenceFrame[startCurrCol + (cuY+lid%cuHeight)*frameWidth]; // At this point, one row of samples is in the shared array. We must reduce it to obtain the redL for each CU
        }
        else if((ctuY+cuY)==0 && (ctuX+cuX)==0){ // CU is in the top-left corner of frame, no reference is available. Fill with predefined DC value
            refL[lid] = valueDC;
        }
        else if((ctuX+cuX)==0 && (ctuY+cuY)>0){ // CU is in the left edge of the frame, we use the top samples to pad the left boundaries
            refL[lid] = referenceFrame[(ctuY+cuY-1)*frameWidth];  // Sample directly above of the first sample inside the CU is padded to left boundary
        }
        bufferGlobalRefL[col][lid%(cuRowsPerCtu*cuHeight)] = refL[lid];

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
        if(lid<(reducedBoundarySize*cuRowsPerCtu*colsPerPass)){ 
            // Subsample top boundary
            int temp = 0;
            int currentRow = (lid/reducedBoundarySize)%cuRowsPerCtu;
            col = firstCol + lid/(reducedBoundarySize*cuRowsPerCtu);

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
                printf("Summed values left [%d] = %d\n", lid, temp);
                printf("Processed values left [%d] = %d\n", lid, (temp + roundingOffset) >> log2DownsamplingFactor);
            }
            //*/     
        }
        barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems computed the subsampled values
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

    // ---------   UNIFIED BUFFERS FOR LEFT BOUNDARIES
    //
    // Move reduced boundary into unified global memory buffer -- ok
    if(lid < reducedBoundarySize*nCusInCtu){
        int idx;
        int nPasses = max(1, (reducedBoundarySize*nCusInCtu)/wgSize); // 
        for(int pass=0; pass<nPasses; pass++){
            // Point to current CTU
            idx = ctuIdx*(ALL_TOTAL_CUS_SizeId12_PER_CTU*BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU*BOUNDARY_SIZE_Id0);
            // Point to current CU size. Even though CUs 4x4 have reducedBoundarySize=2, the ALL_stridedCusPerCtu points to the start of the current CU size and all previous sizes have reducedboundarySize=4
            idx += ALL_stridedCusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY;
            
            unified_redL[idx + pass*wgSize + lid] = bufferGlobalRedL[pass*wgSize + lid];
        }
    }

   // Move complete boundary into global memory
    if(lid < (cuColumnsPerCtu*128)){
        
        int nColumns= cuColumnsPerCtu;
        int nRows = cuRowsPerCtu;

        int columnsPerPass = wgSize/cuHeight;
        int samplesPerRow = cuHeight*cuColumnsPerCtu; // Sum all left boundaries of CUs in the same CU row

        int nPasses = max(1, (nColumns*cuRowsPerCtu*cuHeight) / wgSize);

        int currCuCol, currCuRow, currSample;

        for(int pass=0; pass<nPasses; pass++){
                int stride = pass*wgSize + lid;
                currCuRow = stride / samplesPerRow;
                currCuCol = (stride/cuHeight) % cuColumnsPerCtu;
                currSample = stride % cuHeight;
                // Left boundaries are stored in global memory in CU-raster order. First the boundaries of CU_0, then CU_1, then CU_2, ... Since CUs are upsampled in raster order, this order improves the memory accesses
                unified_refL[ctuIdx*ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES] + ALL_stridedCompleteLeftBoundaries[cuSizeIdx] + currCuRow*cuColumnsPerCtu*cuHeight + currCuCol*cuHeight + currSample] = bufferGlobalRefL[currCuCol][currCuRow*cuHeight + currSample];
        }
    }
}

// This kernel is used to obtain the reduced prediction with all prediction modes
// The prediction of all prediction modes is stored in global memory and returned to the host
// Each WG will process one CTU composed of a single CU size
__kernel void MIP_SizeId2_ALL(__global short *reducedPrediction, const int frameWidth, const int frameHeight, __global long *SAD, __global short* originalSamples, __global short *unified_redT, __global short *unified_redL){
    // Variables for indexing work items and work groups
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    const short ctuIdx = wg/ALL_NUM_CU_SIZES;
    const short cuSizeIdx = wg%ALL_NUM_CU_SIZES;
    const short cuWidth = ALL_widths[cuSizeIdx];
    const short cuHeight = ALL_heights[cuSizeIdx];
    const int nCusInCtu = ALL_cusPerCtu[cuSizeIdx];

    const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
      
    const short cuColumnsPerCtu = ALL_cuColumnsPerCtu[cuSizeIdx];
    const short cuRowsPerCtu = ALL_cuRowsPerCtu[cuSizeIdx];

    // CTU position inside the frame
    const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
    const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

    int boundaryStrideForCtu = ALL_TOTAL_CUS_SizeId12_PER_CTU*BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU*BOUNDARY_SIZE_Id0;
    int currCtuBoundariesIdx = ctuIdx * boundaryStrideForCtu;

    // This buffer stores all predicted CUs inside the current CTU, with a single prediction mode
    // Each CU is processed by 64 workitems, where each workitem conducts the prediction of a single sample
    // When necessary, each workitem will process more than one CU
    // After all CUs are predicted with a single prediction mode, the buffer is moved into global memory and the next prediction mode is tested
    __local short reducedPredictedCtu[ 16384 ]; // When reducedPred=8x8 there are at most 256 CUs per CTU (256*8*8=16384). When reducedPred=4x4 there at exactly 1024 CUs (1024*4*4=16384)

    __local short upsampledPredictedCtu[128*128]; // used to store the entire CTU after upsampling, before computing distortion

    int totalPredictionModes = ALL_numPredModes[cuSizeIdx] + TEST_TRANSPOSED_MODES*ALL_numPredModes[cuSizeIdx];
    const char reducedBoundarySize = ALL_reducedBoundarySizes[cuSizeIdx];
    const char reducedPredSize = ALL_reducedPredSizes[cuSizeIdx];


    // For SizeId=2, each 64 workitems process one CU irrespective of the "original size" since the reduced prediciton is 8x8 (=64) for all of them
    // For SizeId=1, each 16 workitems process one CU irrespective of the "original size" since the reduced prediciton is 4x4 (=16) for all of them
    // For SizeId=0, each 16 workitems process one CU irrespective of the "original size" since the reduced prediciton is 4x4 (=16) for all of them
    const short itemsPerCuInPrediction = reducedPredSize*reducedPredSize;
    char sampleInCu = lid%itemsPerCuInPrediction; // Each workitem processes 1 sample
    int cusPerPass = wgSize/itemsPerCuInPrediction;
    int nPasses = nCusInCtu / cusPerPass;
    short cuIdxInCtu;
    int predSample;
    
    // Compute transposed index inside CU. Used for transposed MIP modes
    char transposedSampleInCu;
    char tempX, tempY;
    tempX = sampleInCu%reducedPredSize;
    tempY = sampleInCu/reducedPredSize;
    transposedSampleInCu = tempX*reducedPredSize + tempY;

    short8 reducedBoundaries, coefficients;
    short4 reducedT, reducedL;
    
    int idx;

    // TODO: Change this for loop to fetch the boundaries a single time for non-transposed and a single time for transposed
    // Use for(transp = [0,1]){ for(mode = [0,5])}
    // TODO: These two for loops will have a different numbe of iterations depending if we are processing SizeId=2 or SizeId=1. This can lead the compiler to generate not optimized code. Consider using he same number of iterations for both SizeIds
    for(int m=0; m<totalPredictionModes; m++){
        
        short mode = m%(totalPredictionModes/2);
        short t = -(m/((totalPredictionModes/2))); // -1 because this value is used in select(), and select() only tests the MSB of the value
        short8 isTransp = (short8) (t,t,t,t,t,t,t,t);

        for(int pass=0; pass<nPasses; pass++){
            cuIdxInCtu = pass*cusPerPass + floor((float)lid/itemsPerCuInPrediction);

            // Point to current CU size. Even though CUs 4x4 have reducedBoundarySize=2, the ALL_stridedCusPerCtu points to the start of the current CU size and all previous sizes have reducedboundarySize=4
            idx = currCtuBoundariesIdx + ALL_stridedCusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY;

            if(reducedBoundarySize==4){
                reducedT = vload4((idx + cuIdxInCtu*reducedBoundarySize)/4, unified_redT);
                reducedL = vload4((idx + cuIdxInCtu*reducedBoundarySize)/4, unified_redL);
                // Create complete boundaries array based on transposed or not-transposed
                reducedBoundaries = select((short8)(reducedT, reducedL), (short8)(reducedL, reducedT), isTransp);    
            }
            else{ // reducedBoundary=2
                reducedT.lo = vload2((idx + cuIdxInCtu*reducedBoundarySize)/2, unified_redT);
                reducedL.lo = vload2((idx + cuIdxInCtu*reducedBoundarySize)/2, unified_redL);
                // Create complete boundaries array based on transposed or not-transposed
                reducedBoundaries.lo = select((short4)(reducedT.lo, reducedL.lo), (short4)(reducedL.lo, reducedT.lo), isTransp.lo);    
            }

            short firstVal = reducedBoundaries.s0;
            // Apply inputOffset to all boundaries except the first, then zero the first. After this the boundaries are ready to be multiplied by the coefficients
            reducedBoundaries = reducedBoundaries - (short8) (firstVal); //(0, firstVal, firstVal, firstVal, firstVal, firstVal, firstVal, firstVal);
            reducedBoundaries.s0 = select((1 << (10-1))-firstVal, 0, reducedPredSize==8);
            

            int offset_lo = dot(convert_float4(reducedBoundaries.lo), convert_float4((short4) (1,1,1,1)));
            int offset_hi = dot(convert_float4(reducedBoundaries.hi), convert_float4((short4) (1,1,1,1)));
            // Compute correct offset based on SizeId (boundarySize)
            int offset = select(offset_lo, offset_lo + offset_hi, reducedBoundarySize==4);            
            
            offset = (1 << (MIP_SHIFT_MATRIX - 1)) - MIP_OFFSET_MATRIX * offset;

            // TODO: Compare performance of naive and dot-product prediction

            // coefficients.s0 = 0;
            // coefficients.s1 = mipMatrix16x16[mode][sampleInCu][0];
            // coefficients.s2 = mipMatrix16x16[mode][sampleInCu][1];
            // coefficients.s3 = mipMatrix16x16[mode][sampleInCu][2];
            // coefficients.s4 = mipMatrix16x16[mode][sampleInCu][3];
            // coefficients.s5 = mipMatrix16x16[mode][sampleInCu][4];
            // coefficients.s6 = mipMatrix16x16[mode][sampleInCu][5];
            // coefficients.s7 = mipMatrix16x16[mode][sampleInCu][6];

            // predSample =  offset;
            // predSample += coefficients.s1 * reducedBoundaries.s1;
            // predSample += coefficients.s2 * reducedBoundaries.s2;
            // predSample += coefficients.s3 * reducedBoundaries.s3;
            // predSample += coefficients.s4 * reducedBoundaries.s4;
            // predSample += coefficients.s5 * reducedBoundaries.s5;
            // predSample += coefficients.s6 * reducedBoundaries.s6;
            // predSample += coefficients.s7 * reducedBoundaries.s7;



            // Fetch the coefficients from global
            uchar8 vectorizedCoeffs;
            if(reducedPredSize==8){ // SizeId==2
                vectorizedCoeffs = vload8(0, &mipMatrix16x16[mode][sampleInCu][0]);
                // Shift the coefficients to the right by 1 element, so that coeff 1 is in position [1]. Zero first coefficient beause it does not exist
                uint8 mask = (uint8)(0,0,1,2,3,4,5,6); 
                vectorizedCoeffs = shuffle(vectorizedCoeffs, mask); 
                vectorizedCoeffs.s0 = 0;
            }
            else if(reducedPredSize==4 && reducedBoundarySize==4){ // SizeId=1
                vectorizedCoeffs = vload8(0, &mipMatrix8x8[mode][sampleInCu][0]);
            }
            else{ // SizeId=0
                vectorizedCoeffs.lo = vload4(0, &mipMatrix4x4[mode][sampleInCu][0]);
            }
            

            // Dot function works with at most 4 values at a time. We must do the lower and higher part individually
            predSample  = offset;
            predSample += dot(convert_float4(reducedBoundaries.lo), convert_float4(vectorizedCoeffs.lo));
            // If reducedBoundary=4 we must add the secodn half of dot product
            if(reducedBoundarySize==4){
                predSample += dot(convert_float4(reducedBoundaries.hi), convert_float4(vectorizedCoeffs.hi));
            }
            
            predSample = (predSample >> MIP_SHIFT_MATRIX) + firstVal;
            predSample = clamp(predSample, 0, (1<<10)-1);

            // Adjust the "correct" position inside the CU depending if the mode is transposed or not            
            short position = select((short) sampleInCu, (short) transposedSampleInCu, t);

            reducedPredictedCtu[cuIdxInCtu*reducedPredSize*reducedPredSize + position] = predSample;

            // Wait until all samples of the CTU are predicted because we will move it into global memory in sequence
            barrier(CLK_LOCAL_MEM_FENCE);

            /*    TRACE PREDICTION
            if(1 && cuIdxInCtu==0 && cuSizeIdx==ALL_AL_32x4 && mode==0 && ctuIdx==16 && lid==0){
                //for(int cu=0; cu<4; cu++){
                    printf("REDUCED PREDICTION: CTU %d, CU %d, Mode %d\n", ctuIdx, cuIdxInCtu, m);
                    printf("Reduced pred size %d\n", reducedPredSize);
                    printf("SUB  Reduced boundaries: %d, %d, %d, %d, %d, %d, %d, %d\n", reducedBoundaries.s0, reducedBoundaries.s1, reducedBoundaries.s2, reducedBoundaries.s3, reducedBoundaries.s4, reducedBoundaries.s5, reducedBoundaries.s6, reducedBoundaries.s7);
                    printf("Coeffs:\n");
                    for(int i=0; i<16; i++){
                        printf("Sample: %d\n  ", i);
                        for(int j=0; j<8; j++){
                            printf("%d,", mipMatrix8x8[mode][i][j]);
                        }
                        printf("\n");
                    }
                    for(int i=0; i<reducedPredSize; i++){
                        for(int j=0; j<reducedPredSize; j++){
                            printf("%d,", reducedPredictedCtu[cuIdxInCtu][i*reducedPredSize+j]);
                        }
                        printf("\n");
                    }
                //}
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/
        }
        // Now we move the prediction from __local into __global memory
        // This points to the start of current CTU in the reduced prediction global buffer
        const int currCtuPredictionIdx = ctuIdx*ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES];
        int currCuPredictionIdx; // This will point to the start of the current CU in the reduced prediction global buffer (i.e., CTU position + offset)
        
        // Should we save the reduced prediction in global memory? If the upsampling is conducted in the same kernel it is not necessary
        int SAVE_REDUCED_PREDICTION_IN_GLOBAL = 1;
        
        if(SAVE_REDUCED_PREDICTION_IN_GLOBAL){
            for(int pass=0; pass<nPasses; pass++){
                cuIdxInCtu = pass*cusPerPass + floor((float)lid/itemsPerCuInPrediction);
                // Point to start of this CU size in global buffer
                currCuPredictionIdx = currCtuPredictionIdx + ALL_stridedPredictionsPerCtu[cuSizeIdx];
                // Point to start of this CU specifically in global buffer
                currCuPredictionIdx += cuIdxInCtu*reducedPredSize*reducedPredSize*ALL_numPredModes[cuSizeIdx]*2;
                // Point to start of the current mode in global buffer
                currCuPredictionIdx += m*reducedPredSize*reducedPredSize;

                reducedPrediction[currCuPredictionIdx + sampleInCu] = reducedPredictedCtu[cuIdxInCtu*reducedPredSize*reducedPredSize + sampleInCu];
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all predicted samples are moved. The next iteration overwrites the local prediction buffer
        }
    } // End of current mode
}

// This kernel is used to fetch the reduced boundaries for all the blocks
// Each WG will process one CTU composed of a single CU size
// It works for square blocks with SizeId=2 (64x64, 32x32, 16x16)
__kernel void initBoundariesSquareSizeId2(__global short *referenceFrame, const int frameWidth, const int frameHeight, __global short *unified_redT, __global short *unified_redL, __global short *unified_refT, __global short *unified_refL){

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

    __local short int bufferGlobalRefT[16][128], bufferGlobalRefL[16][128];

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
        bufferGlobalRefT[row][lid] = refT[lid];
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

    // ---------   UNIFIED BUFFERS FOR TOP BOUNDARIES
    //
    // Move reduced boundary into unified global memory buffer
    if(lid < 4*nCusInCtu){
        int nPasses = max(1, (4*nCusInCtu)/wgSize); // 
        for(int pass=0; pass<nPasses; pass++){
            unified_redT[ctuIdx*TOTAL_CUS_PER_CTU*4 + stridedCusPerCtu[cuSizeIdx]*4 + pass*wgSize + lid] = bufferGlobalRedT[pass*wgSize + lid];
        }
    }

    // Move complete boundary into global memory
    if(lid < (cuRowsPerCtu*128)){
        
        int nRows = cuRowsPerCtu;
        int rowsPerPass = wgSize/128;
        int nPasses = max(1, (nRows*128) / wgSize);

        for(int pass=0; pass<nPasses; pass++){
            int currRow = pass*rowsPerPass + lid/128;
            unified_refT[ctuIdx*stridedCompleteTopBoundaries[NUM_CU_SIZES] + stridedCompleteTopBoundaries[cuSizeIdx] + currRow*128 + lid%128] = bufferGlobalRefT[currRow][lid%128];
        }            
    }

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
        bufferGlobalRefL[col][lid] = refL[lid];

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

    // ---------   UNIFIED BUFFERS FOR LEFT BOUNDARIES
    //
    // Move reduced boundary into unified global memory buffer -- ok
    if(lid < 4*nCusInCtu){
        int nPasses = max(1, (4*nCusInCtu)/wgSize); // 
        for(int pass=0; pass<nPasses; pass++){
            unified_redL[ctuIdx*TOTAL_CUS_PER_CTU*4 + stridedCusPerCtu[cuSizeIdx]*4 + pass*wgSize + lid] = bufferGlobalRedL[pass*wgSize + lid];
        }
    }

   // Move complete boundary into global memory
    if(lid < (cuColumnsPerCtu*128)){
        
        int nColumns= cuColumnsPerCtu;
        int nRows = cuRowsPerCtu;

        int columnsPerPass = wgSize/cuHeight;
        int samplesPerRow = cuHeight*cuColumnsPerCtu; // Sum all left boundaries of CUs in the same CU row

        int nPasses = max(1, (nColumns*128) / wgSize);

        int currCuCol, currCuRow, currSample;

        for(int pass=0; pass<nPasses; pass++){
                int stride = pass*wgSize + lid;
                currCuRow = stride / samplesPerRow;
                currCuCol = (stride/cuHeight) % cuColumnsPerCtu;
                currSample = stride % cuHeight;
                // Left boundaries are stored in global memory in CU-raster order. First the boundaries of CU_0, then CU_1, then CU_2, ... Since CUs are upsampled in raster order, this order improves the memory accesses
                unified_refL[ctuIdx*stridedCompleteLeftBoundaries[NUM_CU_SIZES] + stridedCompleteLeftBoundaries[cuSizeIdx] + currCuRow*cuColumnsPerCtu*cuHeight + currCuCol*cuHeight + currSample] = bufferGlobalRefL[currCuCol][currCuRow*cuHeight + currSample];
        }
    }
}

__kernel void upsampleDistortionSizeId2_ALL(__global short *reducedPrediction, const int frameWidth, const int frameHeight, __global long *SAD, __global long *SATD, __global short* originalSamples, __global short *unified_redT, __global short *unified_redL, __global short *unified_refT, __global short *unified_refL, __global long *minSadHad){
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    const short ctuIdx = wg/NUM_CU_SIZES_SizeId2;
    const short cuSizeIdx = wg%NUM_CU_SIZES_SizeId2;
    const short cuWidth = ALL_widths[cuSizeIdx];
    const short cuHeight = ALL_heights[cuSizeIdx];
    const int nCusInCtu = ALL_cusPerCtu[cuSizeIdx];

    const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
      
    const short cuColumnsPerCtu = ALL_cuColumnsPerCtu[cuSizeIdx];
    const short cuRowsPerCtu = ALL_cuRowsPerCtu[cuSizeIdx];

    // CTU position inside the frame
    const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
    const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

    int boundaryStrideForCtu = ALL_cusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY; // Each CU occupy LARGEST_RED_BOUNDARY (=4) positions in the reduced boundaries buffers
    int currCtuBoundariesIdx = ctuIdx * boundaryStrideForCtu;

    const int numPredictionModes = PREDICTION_MODES_ID2;
    const char reducedBoundarySize = BOUNDARY_SIZE_Id2;
    const char reducedPredSize = REDUCED_PRED_SIZE_Id2;

    const int upsamplingHorizontal = cuWidth / reducedPredSize;
    const int upsamplingVertical = cuHeight / reducedPredSize;

    const int log2UpsamplingHorizontal = (int) log2((float) upsamplingHorizontal);
    const int roundingOffsetHorizontal = 1 << (log2UpsamplingHorizontal - 1);

    const int log2UpsamplingVertical = (int) log2((float) upsamplingVertical);
    const int roundingOffsetVertical = 1 << (log2UpsamplingVertical - 1);
    
    // TODO: Correct this when supporting more block sizes.
    // Correct value is upsamplingHorizontal>1 || upsamplingVertical>1;
    int needUpsampling = 1; 

    // ######################################################################
    //      Variables shared for horizontal and vertical interpolation
    int xPosInCu, yPosInCu, xPosInCtu, yPosInCtu, xPosInFrame, yPosInFrame, idx;
    int valueBefore, valueAfter, beforeIdx, afterIdx;
    int isMiddle;
    int offsetInStride;
    int itemsPerCuInUpsampling;
    int itemsPerCuInFetchOriginal;
    int itemsPerCuInSatd;
    
    // During upsampling, 128 workitems are assigned to conduct the processing of each CU (i.e., with wgSize=256 we process 2 CUs at once)
    // We fetch the boundaries of 1 or 2 CUs depending on wgSize, upsample these CUs with all prediction modes to reuse the boundaries without extra memory access
    // Compute the distortion for these CUs with each prediction mode, then process the next CUs
    __local short localReducedPrediction[2][REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2]; // At most, 2 CUs are processed simultaneously
    __local short localUpsampledPrediction[2][64*64]; // either 1 or 2 CUs are predicted simultaneously, with a maximum dimension of 64x64
    __local short localOriginalSamples[2][64*64];
    __local int localSAD[256], localSATD[256];

    __local int localSadEntireCtu[ALL_MAX_CUS_PER_CTU_SizeId2][PREDICTION_MODES_ID2*2];
    __local int localSatdEntireCtu[ALL_MAX_CUS_PER_CTU_SizeId2][PREDICTION_MODES_ID2*2];

    // Each CU will be upsampled using 128 workitems, irrespective of the CU size
    // We will process 2 CUs simultaneously when wgSize=256
    // CUs with more than 128 samples will require multiple passes (i.e., CUs larger than 8x16 and 16x8)
    itemsPerCuInUpsampling = 128;
    itemsPerCuInFetchOriginal = 128;

    for(int firstCu = 0; firstCu < nCusInCtu; firstCu += wgSize/itemsPerCuInUpsampling){
        int cuIdxInIteration = lid/itemsPerCuInUpsampling; // This represents if the current CU equals firstCU or firstCU+1
        int currCu = firstCu + cuIdxInIteration;

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //      FETCH THE  ORIGINAL SAMPLES FOR THE CUs BEING PROCESSED
        
        int nPassesOriginalFetch = (cuWidth*cuHeight)/itemsPerCuInFetchOriginal;
        int xPosInCu, yPosInCu, xPosInCtu, yPosInCtu, xPosInFrame, yPosInFrame;

        for(int pass=0; pass<nPassesOriginalFetch; pass++){
            idx = pass*itemsPerCuInFetchOriginal + lid%itemsPerCuInFetchOriginal;
            xPosInCu = idx%cuWidth;
            yPosInCu = idx/cuWidth;
            xPosInCtu = ALL_X_POS[cuSizeIdx][currCu] + xPosInCu;
            yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu] + yPosInCu;
            xPosInFrame = ctuX + xPosInCtu;
            yPosInFrame = ctuY + yPosInCtu;
            
            localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] = originalSamples[yPosInFrame*frameWidth + xPosInFrame];
        }
        

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //      FETCH THE BOUNDARIES REQUIRED FOR UPSAMPLING

        // TODO: We only need the reduced left boundary
        __local int refT[2*64], refL[2*64]; // Complete boundaries of the two CUs being processed
        int topBoundariesIdx, leftBoundariesIdx;

        // Points to the current CTU boundaries
        topBoundariesIdx = ctuIdx * ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES];
        leftBoundariesIdx = ctuIdx * ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES];

        // Points to the current CU boundaries
        topBoundariesIdx  += ALL_stridedCompleteTopBoundaries[cuSizeIdx] + currCu*cuWidth;
        leftBoundariesIdx += ALL_stridedCompleteLeftBoundaries[cuSizeIdx] + currCu*cuHeight;
        
        // Fetch TOP boundaries
        if(lid%itemsPerCuInUpsampling < cuWidth){
            refT[cuIdxInIteration*64 + lid%itemsPerCuInUpsampling] =  unified_refT[topBoundariesIdx + lid%itemsPerCuInUpsampling];
        }

        // Fetch LEFT boundaries
        // TODO: We only need the reduced left boundaries. It reduces the number of global memory reads at this point. For the top boundaries we will need the full references
        if(lid%itemsPerCuInUpsampling < cuHeight){
            refL[cuIdxInIteration*64 + lid%itemsPerCuInUpsampling] =  unified_refL[leftBoundariesIdx + lid%itemsPerCuInUpsampling];
        }
        
        /*  TRACE THE BOUNDARIES FOR THE CURRENT CU
        //if(0 && ctuIdx==0 && cuSizeIdx==_16x16 && currCu==45 && lid==128){
        //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && lid%itemsPerCuInUpsampling==0){
        if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && lid%itemsPerCuInUpsampling==0){
            //printf("\n\n\n\n\n OI %d %d\n\n\n\n\n\n", refT[0], refL[0]);
            printf("TOP BOUNDARIES,CTU=%d,WH=%dx%d,CU=%d\n", ctuIdx, cuWidth, cuHeight, currCu);
            for(int i=0; i<cuWidth; i++){
                printf("%d,", refT[(currCu%2)*64 + i]);
            }
            printf("\n");
            printf("LEFT BOUNDARIES,CTU=%d,WH=%dx%d,CU=%d\n", ctuIdx, cuWidth, cuHeight, currCu);
            for(int i=0; i<cuHeight; i++){
                printf("%d,", refL[(currCu%2)*64 + i]);
            }
            printf("\n");
        }
        //*/

        int idxCurrCuAndMode = 0;
        // Now we do the upsampling for all prediction modes of the current 2 CUs
        for(int mode=0; mode<numPredictionModes*2; mode++){
            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      FETCH THE REDUCED PREDICTION FOR CURRENT MODE AND CUs
    
            // Point to the start of current CTU
            idxCurrCuAndMode = ctuIdx*ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES];
            // Point to start of current CU size
            idxCurrCuAndMode += ALL_stridedPredictionsPerCtu[cuSizeIdx];
            // Point to start of current CU
            idxCurrCuAndMode += currCu*reducedPredSize*reducedPredSize*numPredictionModes*2;
            // Point to start of current prediction mode
            idxCurrCuAndMode += mode*reducedPredSize*reducedPredSize;

            if(lid%itemsPerCuInUpsampling < (reducedPredSize*reducedPredSize)){
                localReducedPrediction[cuIdxInIteration][lid%itemsPerCuInUpsampling] = reducedPrediction[idxCurrCuAndMode + lid%itemsPerCuInUpsampling];
            }

            barrier(CLK_LOCAL_MEM_FENCE); // Wait until the whole buffer for reduced prediction is filled


            /* Trace reduced prediction fetched from global memory
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && mode==0 && lid%itemsPerCuInUpsampling==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && mode==0 && lid%itemsPerCuInUpsampling==0){
                printf("REDUCED PREDICTION,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
                for(int i=0; i<8; i++){
                    for(int j=0; j<8; j++){
                        printf("%d,", localReducedPrediction[cuIdxInIteration][i*8+j]);
                    }
                    printf("\n");
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/


            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      START WITH HORIZONTAL UPSAMPLING...

            // TODO: This should be corrected when we process CUs with less samples than 16x16 (i.e., 16x8 and 8x16)
            int nPassesHorizontalUpsampling = max(1, (cuWidth*reducedPredSize)/itemsPerCuInUpsampling);

            for(int pass=0; pass<nPassesHorizontalUpsampling; pass++){
                int idx = pass*itemsPerCuInUpsampling + lid%itemsPerCuInUpsampling;
                xPosInCu = idx%cuWidth;
                yPosInCu = (idx/cuWidth)*upsamplingVertical + upsamplingVertical-1;
                xPosInCtu = ALL_X_POS[cuSizeIdx][currCu]; // (currCu%cuColumnsPerCtu)*cuWidth + xPosInCu;
                yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu]; // (currCu/cuColumnsPerCtu)*cuHeight + yPosInCu;

                isMiddle = xPosInCu>=upsamplingHorizontal; // In this case, the left boundary is not used
                offsetInStride = xPosInCu%upsamplingHorizontal+1; // Position inside one window where samples are being interpolated. BeforeReference has stride=0, first interpolated sample has stride=1            }

                if(lid%itemsPerCuInUpsampling < cuWidth*reducedPredSize){ // For CUs 8x16 the horizontal upsampling is 1 and only 64 workitems work, the others are idle (in this case there is not upsampling, only copying)
                    // For the first couple of sample columns, the "before" reference is the refL buffer
                    if(isMiddle == 0){
                        // Predicted value that is before the current sample
                        valueBefore = refL[cuIdxInIteration*64 + yPosInCu];
                        // Predicted value that is after the current sample
                        valueAfter = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*reducedPredSize + (xPosInCu>>log2UpsamplingHorizontal)];
                    }
                    else{ // isMiddle == 1
                        valueBefore = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*reducedPredSize + (xPosInCu>>log2UpsamplingHorizontal) - 1];
                        valueAfter  = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*reducedPredSize + (xPosInCu>>log2UpsamplingHorizontal)];
                    }

                    int filteredSample = ((upsamplingHorizontal-offsetInStride)*valueBefore + offsetInStride*valueAfter + roundingOffsetHorizontal)>>log2UpsamplingHorizontal;
                    localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth+xPosInCu] = filteredSample;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish the horizontal upsampling

            /* TRACE HORIZONTAL UPSAMPLING RESULT
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && mode==0 && lid%itemsPerCuInUpsampling==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && mode==0 && lid%itemsPerCuInUpsampling==0){
                printf("HORIZONTALLY-UPSAMPLED PREDICTION,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
                for(int i=0; i<8; i++){
                // for(int i=0; i<60; i++){
                    for(int j=0; j<cuWidth; j++){
                        printf("%d,", localUpsampledPrediction[cuIdxInIteration][(i*upsamplingVertical + upsamplingVertical-1)*cuWidth + j]);
                        // printf("%d,", localUpsampledPrediction[cuIdxInIteration][i*cuWidth + j]);
                    }
                    printf("\n");
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
            //*/

            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      PROCEED WITH VERTICAL UPSAMPLING...

            int nPassesVerticalUpsampling = (cuWidth*cuHeight)/itemsPerCuInUpsampling;

            for(int pass=0; pass<nPassesVerticalUpsampling; pass++){
                int idx = pass*itemsPerCuInUpsampling + lid%itemsPerCuInUpsampling;
                xPosInCu = idx%cuWidth;
                yPosInCu = (idx/cuWidth);
                xPosInCtu = ALL_X_POS[cuSizeIdx][currCu]; // (currCu%cuColumnsPerCtu)*cuWidth + xPosInCu;
                yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu]; // (currCu/cuColumnsPerCtu)*cuHeight + yPosInCu;

                isMiddle = yPosInCu>=upsamplingVertical; // In this case, the top boundary is not used
                offsetInStride = yPosInCu%upsamplingVertical+1; // Position inside one window where samples are being interpolated. BeforeReference has stride=0, first interpolated sample has stride=1            }

                // For the first couple of sample columns, the "before" reference is the top boundaries buffer
                if(isMiddle == 0){
                    // Predicted value that is before the current sample
                    valueBefore = refT[cuIdxInIteration*64 + xPosInCu];
                    // Predicted value that is after the current sample
                    valueAfter = localUpsampledPrediction[cuIdxInIteration][(((yPosInCu>>log2UpsamplingVertical)<<log2UpsamplingVertical) + upsamplingVertical-1)*cuWidth + xPosInCu];
                }
                else{ // isMiddle == 1
                    valueBefore = localUpsampledPrediction[cuIdxInIteration][(((yPosInCu>>log2UpsamplingVertical)<<log2UpsamplingVertical)-1)*cuWidth + xPosInCu];
                    valueAfter  = localUpsampledPrediction[cuIdxInIteration][(((yPosInCu>>log2UpsamplingVertical)<<log2UpsamplingVertical)+upsamplingVertical-1)*cuWidth + xPosInCu];
                }

                int filteredSample = ((upsamplingVertical-offsetInStride)*valueBefore + offsetInStride*valueAfter + roundingOffsetVertical)>>log2UpsamplingVertical;
                localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth+xPosInCu] = filteredSample;                
            }

            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish the vertical upsampling

            /* TRACE COMPLETE UPSAMPLING
            // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && currCu==3 && mode==0 && lid%itemsPerCuInUpsampling==0){
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && mode==0 && lid%itemsPerCuInUpsampling==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && mode==0 && lid%itemsPerCuInUpsampling==0){
                printf("UPSAMPLED PREDICTION,CTU=%d,WH=%dx%d,CU=%d\n", ctuIdx, cuWidth, cuHeight, currCu);
                for(int i=0; i<cuHeight; i++){
                    for(int j=0; j<cuWidth; j++){
                        printf("%d,", localUpsampledPrediction[cuIdxInIteration][i*cuWidth+j]);
                    }
                    printf("\n");
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/

            
            // At this point the upsampling for the current mode is complete. We can compute the distortion...
            
            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      COMPUTE SAD FOR THE CURRENT CU

            // Here, each workitem will compute the SAD at one or more position, and accumulate the result in localSAD[lid]
            // At the end we must reduce the valeus between 0-127 and 128-255 to obtain the final SAD of each CU
            localSAD[lid] = 0;
            localSATD[lid] = 0;

            int nPassesForSad = nPassesOriginalFetch;

            int NEW_SATD = 0;
            int CONDUCT_SATD = 1;

            for(int pass=0; pass<nPassesForSad; pass++){
                idx = pass*itemsPerCuInFetchOriginal + lid%itemsPerCuInFetchOriginal;
                xPosInCu = idx%cuWidth;
                yPosInCu = idx/cuWidth;
                
                localSAD[lid] += (int) abs(localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] - localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu]);
                // Substitute predicted samples by prediction error
                if(NEW_SATD)
                    localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] = localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] - localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu];
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish their SAD computation
            

            // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && currCu==3 && lid%128==0 && mode==0){
            //     printf("SERIAL SAD FOR,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
            //     for(int i=0; i<128; i++){
            //         printf("%d,", localSAD[lid+i]);
            //     }
            //     printf("\n");
            // }

            // PARALLEL REDUCTION
            int nPassesParallelSum = 7; // log2(128)
            int stride = 64;
            int baseIdx = (lid/128)*128 + lid%128;
            for(int pass=0; pass<nPassesParallelSum; pass++){
                if(lid%128 < stride){
                    localSAD[baseIdx] = localSAD[baseIdx] + localSAD[baseIdx+stride];
                    stride = stride/2;
                }    
                barrier(CLK_LOCAL_MEM_FENCE);  
            }

            /* TRACE SAD
            if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && currCu==0 && lid%128==0){
                printf("SAD,CTU=%d,WH=%dx%d,CU=%d,MODE=%d,SAD=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode, localSAD[(cuIdxInIteration%2)*128]);
                // printf("OI,localSAD[lid]=%d\n", localSAD[lid]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/      
            
            localSATD[lid] = 0;

            // ALLOW SKIPPING SATD TO MEASURE PROCESSING TIME
            if(CONDUCT_SATD){
                // TODO: This NEW_SATD method WAS NOT improved to support the remaining block sizes and alignments
                if(NEW_SATD){
                    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                    //
                    //      ALTERNATIVE SATD COMPUTATION
                    int subblocksPerCu = (cuWidth*cuHeight)/16;
                    int subblocksColumnsPerCu = cuWidth/4;
                    int nPasses = (cuWidth*cuHeight)/128;
                    
                    // TODO: Transform this section into a function
                    // Compute transformed differences at the 4x4-level
                    for(int pass=0; pass<nPasses; pass++){
                        int idx = pass*128 + lid%128;
                        int currSubblock = idx/16;
                        int subblockX = (currSubblock%subblocksColumnsPerCu)*4;
                        int subblockY = (currSubblock/subblocksColumnsPerCu)*4;
                        // Position of the sample that will be computed
                        int i = idx%16;
                        int i_xInCu = subblockX + i%4;
                        int i_yInCu = subblockY + i/4;
                        int i_idxInCu = i_yInCu*cuWidth + i_xInCu;
                        // Used to index the arguments of the SATD for one position, and store the temporary value before updating the array
                        int j, j_xInCu, j_yInCu, k, k_xInCu, k_yInCu, f_i;
                        int j_idxInCu, k_idxInCu;
                        int tempValue;

                        // Now the hadamard transform is applied to the 4x4 blocks. We derive the index of operands and operation (+/-) based on the index of current coefficient
                        // m[i] = diff[j] +/-(f_i) diff[k]
                        // The values of j, f_i and k are computed based on i

                        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                        // 
                        // 1st
                        j = i - 4*(i>=8) - 8*(i>=12);
                        k = 12 + i - 8*(i>=4) - 4*(i>=8);
                        f_i = i<8 ? 1 : -1;
                        
                        j_xInCu = subblockX + j%4;
                        j_yInCu = subblockY + j/4;
                        j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                        k_xInCu = subblockX + k%4;
                        k_yInCu = subblockY + k/4;
                        k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                        // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                        tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                        localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                        // 
                        // 2nd
                        j = i + 4*(i>=4) - 12*(i>=8) + 8*(i>=12);
                        k = 4 + i + 4*(i>=4) - 12*(i>=8);
                        f_i = i<8 ? 1 : -1;
                        
                        j_xInCu = subblockX + j%4;
                        j_yInCu = subblockY + j/4;
                        j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                        k_xInCu = subblockX + k%4;
                        k_yInCu = subblockY + k/4;
                        k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                        // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                        tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                        localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                        // barrier(CLK_LOCAL_MEM_FENCE);                


                        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                        // 
                        // 3rd
                        j = 4*(i/4) + 1*(i%4==1) + 1*(i%4==2);
                        k = 3 + 4*(i/4) - 1*(i%4==1) - 1*(i%4==2);
                        f_i = i%4<2 ? 1 : -1;
                        
                        j_xInCu = subblockX + j%4;
                        j_yInCu = subblockY + j/4;
                        j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                        k_xInCu = subblockX + k%4;
                        k_yInCu = subblockY + k/4;
                        k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                        // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                        tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                        localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                        // barrier(CLK_LOCAL_MEM_FENCE);  

                        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                        // 
                        // 4th
                        j = 4*(i/4) + 2*(i%4==2) + 3*(i%4==3);
                        k = 1 + 4*(i/4) + 2*(i%4==2) + 1*(i%4==3);
                        f_i = i%2==0 ? 1 : -1;
                        
                        j_xInCu = subblockX + j%4;
                        j_yInCu = subblockY + j/4;
                        j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                        k_xInCu = subblockX + k%4;
                        k_yInCu = subblockY + k/4;
                        k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                        // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                        tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // TODO: MAYBE it is not necessary to synch the workitems here because the next iteration will work over a different set of values. Only synch outside the for loop
                        // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                        localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                        barrier(CLK_LOCAL_MEM_FENCE);  
                    }


                    //* 
                    // REDUCE TOTAL SATD IN PARALLEL
                    // At this point we have the transformed differences for all subblocks 4x4.
                    // Now each workitem will sum the absolute coefficients and save in its position in localSATD[lid]
                    // Then, these values are reduced
                    itemsPerCuInSatd = min(128, (cuWidth*cuHeight)/16);
                    int nPassesSumCoefficients = max(1, subblocksPerCu/itemsPerCuInSatd);
                    if(lid%128<itemsPerCuInSatd){
                        int sb, subblockX, subblockY, startRow, currSATD;
                        for(int pass=0; pass<nPassesSumCoefficients; pass++){
                            sb = pass*itemsPerCuInSatd + lid%128; // Subblock this workitem will process
                            currSATD = 0;
                            subblockX = (sb%subblocksColumnsPerCu)*4;
                            subblockY = (sb/subblocksColumnsPerCu)*4;
                            startRow = subblockY*cuWidth + subblockX;
                            for(int row=0; row<4; row++){
                                for(int col=0; col<4; col++){
                                    currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow + row*cuWidth + col]);
                                }
                            }
                            currSATD -= abs(localUpsampledPrediction[cuIdxInIteration][startRow]);
                            currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow]) >> 2;
                            currSATD = ((currSATD+1)>>1);
                            
                            // if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && currCu==0 && mode==0){// } && lid%128==0){
                            //     printf("CTU=%d,cuSize=%d,cuIdx=%d,sb=%d,SATD=%d\n", ctuIdx, cuSizeIdx, currCu, sb, currSATD);
                            // }


                            localSATD[lid] += currSATD;
                            currSATD = 0;
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);  // Wait until all partial SATDs are computed
                    //*/

                    //*
                    // PARALLEL REDUCTION
                    nPassesParallelSum = (int) log2( (float) min(128, cuWidth*cuHeight/16) ); // log2()
                    stride = itemsPerCuInSatd/2;
                    baseIdx = (lid/128)*128 + lid%128;
                    for(int pass=0; pass<nPassesParallelSum; pass++){
                        if(lid%128 < stride){
                            localSATD[baseIdx] = localSATD[baseIdx] + localSATD[baseIdx+stride];
                            stride = stride/2;
                        }    
                        barrier(CLK_LOCAL_MEM_FENCE);  
                    }
                    //*/
                    
                    
                    /*
                    // Sum individual values and reduce SATD in sequential fashion
                    if(lid%128 == 0){
                        int currSATD = 0, totalSATD = 0;
                        for(int sb=1; sb<subblocksPerCu; sb++){
                            localSATD[lid] += localSATD[lid+sb];
                            // int subblockX = (sb%subblocksColumnsPerCu)*4;
                            // int subblockY = (sb/subblocksColumnsPerCu)*4;
                            // int startRow = subblockY*cuWidth + subblockX;
                            // for(int row=0; row<4; row++){
                            //     for(int col=0; col<4; col++)
                            //         currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow + row*cuWidth + col]);
                            // }
                            // currSATD -= abs(localUpsampledPrediction[cuIdxInIteration][startRow]);
                            // currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow]) >> 2;
                            // currSATD = (currSATD+1)>>1;
                            
                            // totalSATD += currSATD;

                            // // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && mode==11 && currCu==2 && lid%128==0){
                            // //     // printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %ld, Total SATD %ld\n", lid, idx, subblockX, subblockY, currSATD, totalSATD);
                            // //     printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %d, Total SATD %d\n", lid, idx, subblockX, subblockY, currSATD, totalSATD);
                            // // }
                            // currSATD = 0;
                        }
                        // localSATD[lid] = totalSATD;
                    }
                    //*/
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                
                
                if(!NEW_SATD){
                    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                    //
                    //      COMPUTE SATD FOR THE CURRENT CU
                    //*
                    int idxSubblock;
                    int nPassesForSatd = max(1,nPassesOriginalFetch/16);
                    itemsPerCuInSatd = min(128, (cuWidth*cuHeight)/16);
                    int subblockX, subblockY;
                    short16 origSubblock, predSubblock;

                    // Here, each workitem will compute the SATD for one or more subblocks 4x4, and accumulate the results in __local localSATD
                    if((lid%128) < itemsPerCuInSatd){
                        for(int pass=0; pass<nPassesForSatd; pass++){
                            idxSubblock = pass*itemsPerCuInSatd + lid%128;
                            subblockX = (idxSubblock%(cuWidth/4))<<2;
                            subblockY = (idxSubblock/(cuWidth/4))<<2;
                            idx = subblockY*(cuWidth/4) + subblockX/4;


                            // 1st row                    
                            origSubblock.lo.lo = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.lo.lo = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                            // 2nd row
                            idx += cuWidth/4;
                            origSubblock.lo.hi = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.lo.hi = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                            // 3rd row
                            idx += cuWidth/4;
                            origSubblock.hi.lo = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.hi.lo = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                            // 4th row
                            idx += cuWidth/4;
                            origSubblock.hi.hi = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.hi.hi = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);

                            localSATD[lid] += satd_4x4(origSubblock, predSubblock);
                            
                            /* TRACE INTERMEDIARY SATD 4X4
                            if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && mode==0 && currCu==0){//} && lid==0){
                                printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %ld\n", lid, idx, subblockX, subblockY, satd_4x4(origSubblock, predSubblock));
                                // printf("Orig subblock\n");
                                // printf("%d,%d,%d,%d\n", origSubblock.s0, origSubblock.s1, origSubblock.s2, origSubblock.s3);
                                // printf("%d,%d,%d,%d\n", origSubblock.s4, origSubblock.s5, origSubblock.s6, origSubblock.s7);
                                // printf("%d,%d,%d,%d\n", origSubblock.s8, origSubblock.s9, origSubblock.sa, origSubblock.sb);
                                // printf("%d,%d,%d,%d\n", origSubblock.sc, origSubblock.sd, origSubblock.se, origSubblock.sf);
                            
                                // printf("Pred subblock\n");
                                // printf("%d,%d,%d,%d\n", predSubblock.s0, predSubblock.s1, predSubblock.s2, predSubblock.s3);
                                // printf("%d,%d,%d,%d\n", predSubblock.s4, predSubblock.s5, predSubblock.s6, predSubblock.s7);
                                // printf("%d,%d,%d,%d\n", predSubblock.s8, predSubblock.s9, predSubblock.sa, predSubblock.sb);
                                // printf("%d,%d,%d,%d\n\n\n", predSubblock.sc, predSubblock.sd, predSubblock.se, predSubblock.sf);                    
                            }
                            //*/
                            
                            //*
                        }
                    }

                    barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish their SATD computation

                    // PARALLEL REDUCTION
                    nPassesParallelSum = (int) log2( (float) min(128, cuWidth*cuHeight/16) ); // log2()
                    stride = itemsPerCuInSatd/2;
                    baseIdx = (lid/128)*128 + lid%128;
                    for(int pass=0; pass<nPassesParallelSum; pass++){
                        if(lid%128 < stride){
                            localSATD[baseIdx] = localSATD[baseIdx] + localSATD[baseIdx+stride];
                            stride = stride/2;
                        }    
                        barrier(CLK_LOCAL_MEM_FENCE);  
                    }
                }
            }


            //*/

            // Save SAD and SATD of current CU/mode in a __local buffer. We only access global memory when all SAD values are computed or all CUs
            if(lid==0){
                localSadEntireCtu[firstCu][mode] = localSAD[0];
                localSadEntireCtu[firstCu+1][mode] = localSAD[128];

                localSatdEntireCtu[firstCu][mode] = localSATD[0];
                localSatdEntireCtu[firstCu+1][mode] = localSATD[128];
            } 

        } // Finish current mode
    } // Finish current pair of CUs   
    
    // When all CUs are processed, we move the results into global buffer
    if(lid < ALL_cusPerCtu[cuSizeIdx]*numPredictionModes*2){
        int nPassesMoveSadIntoGlobal = max(1, (int)ceil((1.0*ALL_cusPerCtu[cuSizeIdx]*numPredictionModes*2)/wgSize));
        int idxInLocal, cu, mode, idxInGlobal;
        
        for(int pass=0; pass<nPassesMoveSadIntoGlobal; pass++){
            idxInLocal = pass*wgSize + lid;
            cu = idxInLocal / (numPredictionModes*2);
            mode = idxInLocal % (numPredictionModes*2);

            // Point to current CTU in global buffer
            idxInGlobal = ctuIdx*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES];
            // Point to current CU size in global buffer
            idxInGlobal    += ALL_stridedDistortionsPerCtu[cuSizeIdx];
            // Point to current CU and mode in global buffer
            idxInGlobal    += cu*numPredictionModes*2 + mode;

            if(cu < ALL_cusPerCtu[cuSizeIdx]){
                SAD[idxInGlobal] = ( long ) localSadEntireCtu[cu][mode];
                SATD[idxInGlobal] = ( long ) localSatdEntireCtu[cu][mode];
                minSadHad[idxInGlobal] = (long) min(2*localSadEntireCtu[cu][mode], localSatdEntireCtu[cu][mode]);
            }
        }
    }
}

__kernel void upsampleDistortionSizeId1_ALL(__global short *reducedPrediction, const int frameWidth, const int frameHeight, __global long *SAD, __global long *SATD, __global short* originalSamples, __global short *unified_redT, __global short *unified_redL, __global short *unified_refT, __global short *unified_refL, __global long *minSadHad){
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    const short ctuIdx = wg/NUM_CU_SIZES_SizeId1;
    const short cuSizeIdx = wg%NUM_CU_SIZES_SizeId1 + FIRST_SizeId1;
    const short cuWidth = ALL_widths[cuSizeIdx];
    const short cuHeight = ALL_heights[cuSizeIdx];
    const int nCusInCtu = ALL_cusPerCtu[cuSizeIdx];

    const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
      
    const short cuColumnsPerCtu = ALL_cuColumnsPerCtu[cuSizeIdx];
    const short cuRowsPerCtu = ALL_cuRowsPerCtu[cuSizeIdx];

    // CTU position inside the frame
    const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
    const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

    int boundaryStrideForCtu = ALL_cusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY; // Each CU occupy LARGEST_RED_BOUNDARY (=4) positions in the reduced boundaries buffers
    int currCtuBoundariesIdx = ctuIdx * boundaryStrideForCtu;

    const int numPredictionModes = PREDICTION_MODES_ID1;
    const char reducedBoundarySize = BOUNDARY_SIZE_Id1;
    const char reducedPredSize = REDUCED_PRED_SIZE_Id1;

    const int upsamplingHorizontal = cuWidth / reducedPredSize;
    const int upsamplingVertical = cuHeight / reducedPredSize;

    const int log2UpsamplingHorizontal = (int) log2((float) upsamplingHorizontal);
    const int roundingOffsetHorizontal = 1 << (log2UpsamplingHorizontal - 1);

    const int log2UpsamplingVertical = (int) log2((float) upsamplingVertical);
    const int roundingOffsetVertical = 1 << (log2UpsamplingVertical - 1);
    
    // TODO: Correct this when supporting more block sizes.
    // Correct value is upsamplingHorizontal>1 || upsamplingVertical>1;
    int needUpsampling = 1; 

    // ######################################################################
    //      Variables shared for horizontal and vertical interpolation
    int xPosInCu, yPosInCu, xPosInCtu, yPosInCtu, xPosInFrame, yPosInFrame, idx;
    int valueBefore, valueAfter, beforeIdx, afterIdx;
    int isMiddle;
    int offsetInStride;
    int itemsPerCuInUpsampling;
    int itemsPerCuInFetchOriginal;
    int itemsPerCuInSatd;
    
    // During upsampling, 32 workitems are assigned to conduct the processing of each CU (i.e., with wgSize=256 we process 8 CUs at once) This is based on the size of the smaller CU 4x8=32
    // We fetch the boundaries of CUs depending on wgSize, upsample these CUs with all prediction modes to reuse the boundaries without extra memory access
    // Compute the distortion for these CUs with each prediction mode, then process the next CUs
    __local short localReducedPrediction[8][REDUCED_PRED_SIZE_Id1*REDUCED_PRED_SIZE_Id1]; // At most, 8 CUs are processed simultaneously
    __local short localUpsampledPrediction[8][32*4]; // 8 CUs are predicted simultaneously, with a maximum dimension of 32x4 or 4x32
    __local short localOriginalSamples[8][32*4];
    __local int localSAD[256], localSATD[256];

    __local int localSadEntireCtu[ALL_MAX_CUS_PER_CTU_SizeId1][PREDICTION_MODES_ID1*2];
    __local int localSatdEntireCtu[ALL_MAX_CUS_PER_CTU_SizeId1][PREDICTION_MODES_ID1*2];

    // Each CU will be upsampled using 32 workitems, irrespective of the CU size
    // We will process 8 CUs simultaneously when wgSize=256
    // CUs with more than 32 samples will require multiple passes (i.e., CUs larger than 8x16 and 16x8)
    itemsPerCuInUpsampling = 32;
    itemsPerCuInFetchOriginal = 32;

    for(int firstCu = 0; firstCu < nCusInCtu; firstCu += wgSize/itemsPerCuInUpsampling){
        int cuIdxInIteration = lid/itemsPerCuInUpsampling; // This represents if the current CU equals firstCU, firstCU+1, firstCU+2, ...
        int currCu = firstCu + cuIdxInIteration;

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //      FETCH THE  ORIGINAL SAMPLES FOR THE CUs BEING PROCESSED
        
        int nPassesOriginalFetch = (cuWidth*cuHeight)/itemsPerCuInFetchOriginal;
        int xPosInCu, yPosInCu, xPosInCtu, yPosInCtu, xPosInFrame, yPosInFrame;

        for(int pass=0; pass<nPassesOriginalFetch; pass++){
            idx = pass*itemsPerCuInFetchOriginal + lid%itemsPerCuInFetchOriginal;
            xPosInCu = idx%cuWidth;
            yPosInCu = idx/cuWidth;
            xPosInCtu = ALL_X_POS[cuSizeIdx][currCu] + xPosInCu;
            yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu] + yPosInCu;
            xPosInFrame = ctuX + xPosInCtu;
            yPosInFrame = ctuY + yPosInCtu;
            
            localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] = originalSamples[yPosInFrame*frameWidth + xPosInFrame];
        }
        

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //      FETCH THE BOUNDARIES REQUIRED FOR UPSAMPLING

        // TODO: We only need the reduced left boundary
        __local int refT[8*32], refL[8*32]; // Complete boundaries of the 8 CUs being processed
        int topBoundariesIdx, leftBoundariesIdx;

        // Points to the current CTU boundaries
        topBoundariesIdx = ctuIdx * ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES];
        leftBoundariesIdx = ctuIdx * ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES];

        // Points to the current CU boundaries
        topBoundariesIdx  += ALL_stridedCompleteTopBoundaries[cuSizeIdx] + currCu*cuWidth;
        leftBoundariesIdx += ALL_stridedCompleteLeftBoundaries[cuSizeIdx] + currCu*cuHeight;
        
        // Fetch TOP boundaries
        if(lid%itemsPerCuInUpsampling < cuWidth){
            refT[cuIdxInIteration*32 + lid%itemsPerCuInUpsampling] =  unified_refT[topBoundariesIdx + lid%itemsPerCuInUpsampling];
        }

        // Fetch LEFT boundaries
        // TODO: We only need the reduced left boundaries. It reduces the number of global memory reads at this point. For the top boundaries we will need the full references
        if(lid%itemsPerCuInUpsampling < cuHeight){
            refL[cuIdxInIteration*32 + lid%itemsPerCuInUpsampling] =  unified_refL[leftBoundariesIdx + lid%itemsPerCuInUpsampling];
        }
        
        /*  TRACE THE BOUNDARIES FOR THE CURRENT CU
        //if(0 && ctuIdx==0 && cuSizeIdx==_16x16 && currCu==45 && lid==128){
        //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && lid%itemsPerCuInUpsampling==0){
        if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && lid%itemsPerCuInUpsampling==0){
            //printf("\n\n\n\n\n OI %d %d\n\n\n\n\n\n", refT[0], refL[0]);
            printf("TOP BOUNDARIES,CTU=%d,WH=%dx%d,CU=%d\n", ctuIdx, cuWidth, cuHeight, currCu);
            for(int i=0; i<cuWidth; i++){
                printf("%d,", refT[(currCu%8)*32 + i]);
            }
            printf("\n");
            printf("LEFT BOUNDARIES,CTU=%d,WH=%dx%d,CU=%d\n", ctuIdx, cuWidth, cuHeight, currCu);
            for(int i=0; i<cuHeight; i++){
                printf("%d,", refL[(currCu%8)*32 + i]);
            }
            printf("\n");
        }
        //*/

        int idxCurrCuAndMode = 0;
        // Now we do the upsampling for all prediction modes of the current 8 CUs
        for(int mode=0; mode<numPredictionModes*2; mode++){
            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      FETCH THE REDUCED PREDICTION FOR CURRENT MODE AND CUs
    
            // Point to the start of current CTU
            idxCurrCuAndMode = ctuIdx*ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES];
            // Point to start of current CU size
            idxCurrCuAndMode += ALL_stridedPredictionsPerCtu[cuSizeIdx];
            // Point to start of current CU
            idxCurrCuAndMode += currCu*reducedPredSize*reducedPredSize*numPredictionModes*2;
            // Point to start of current prediction mode
            idxCurrCuAndMode += mode*reducedPredSize*reducedPredSize;

            if(lid%itemsPerCuInUpsampling < (reducedPredSize*reducedPredSize)){
                localReducedPrediction[cuIdxInIteration][lid%itemsPerCuInUpsampling] = reducedPrediction[idxCurrCuAndMode + lid%itemsPerCuInUpsampling];
            }

            barrier(CLK_LOCAL_MEM_FENCE); // Wait until the whole buffer for reduced prediction is filled


            /* Trace reduced prediction fetched from global memory
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && mode==0 && lid%itemsPerCuInUpsampling==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && mode==0 && lid%itemsPerCuInUpsampling==0){
                printf("REDUCED PREDICTION,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
                for(int i=0; i<4; i++){
                    for(int j=0; j<4; j++){
                        printf("%d,", localReducedPrediction[cuIdxInIteration][i*4+j]);
                    }
                    printf("\n");
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/


            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      START WITH HORIZONTAL UPSAMPLING...

            // TODO: This should be corrected when we process CUs with less samples than 16x16 (i.e., 16x8 and 8x16)
            int nPassesHorizontalUpsampling = max(1, (cuWidth*reducedPredSize)/itemsPerCuInUpsampling);

            for(int pass=0; pass<nPassesHorizontalUpsampling; pass++){
                int idx = pass*itemsPerCuInUpsampling + lid%itemsPerCuInUpsampling;
                xPosInCu = idx%cuWidth;
                yPosInCu = (idx/cuWidth)*upsamplingVertical + upsamplingVertical-1;
                xPosInCtu = ALL_X_POS[cuSizeIdx][currCu]; // (currCu%cuColumnsPerCtu)*cuWidth + xPosInCu;
                yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu]; // (currCu/cuColumnsPerCtu)*cuHeight + yPosInCu;

                isMiddle = xPosInCu>=upsamplingHorizontal; // In this case, the left boundary is not used
                offsetInStride = xPosInCu%upsamplingHorizontal+1; // Position inside one window where samples are being interpolated. BeforeReference has stride=0, first interpolated sample has stride=1            }

                if(lid%itemsPerCuInUpsampling < cuWidth*reducedPredSize){ // For CUs 4x8 the horizontal upsampling is 1 and only 16 workitems work, the others are idle (in this case there is not upsampling, only copying)
                    // For the first couple of sample columns, the "before" reference is the refL buffer
                    if(isMiddle == 0){
                        // Predicted value that is before the current sample
                        valueBefore = refL[cuIdxInIteration*32 + yPosInCu];
                        // Predicted value that is after the current sample
                        valueAfter = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*reducedPredSize + (xPosInCu>>log2UpsamplingHorizontal)];
                    }
                    else{ // isMiddle == 1
                        valueBefore = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*reducedPredSize + (xPosInCu>>log2UpsamplingHorizontal) - 1];
                        valueAfter  = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*reducedPredSize + (xPosInCu>>log2UpsamplingHorizontal)];
                    }

                    int filteredSample = ((upsamplingHorizontal-offsetInStride)*valueBefore + offsetInStride*valueAfter + roundingOffsetHorizontal)>>log2UpsamplingHorizontal;
                    localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth+xPosInCu] = filteredSample;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish the horizontal upsampling

            /* TRACE HORIZONTAL UPSAMPLING RESULT
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && mode==0 && lid%itemsPerCuInUpsampling==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && mode==0 && lid%itemsPerCuInUpsampling==0){
                printf("HORIZONTALLY-UPSAMPLED PREDICTION,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
                for(int i=0; i<8; i++){
                // for(int i=0; i<60; i++){
                    for(int j=0; j<cuWidth; j++){
                        printf("%d,", localUpsampledPrediction[cuIdxInIteration][(i*upsamplingVertical + upsamplingVertical-1)*cuWidth + j]);
                        // printf("%d,", localUpsampledPrediction[cuIdxInIteration][i*cuWidth + j]);
                    }
                    printf("\n");
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE); 
            //*/

            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      PROCEED WITH VERTICAL UPSAMPLING...

            int nPassesVerticalUpsampling = (cuWidth*cuHeight)/itemsPerCuInUpsampling;

            for(int pass=0; pass<nPassesVerticalUpsampling; pass++){
                int idx = pass*itemsPerCuInUpsampling + lid%itemsPerCuInUpsampling;
                xPosInCu = idx%cuWidth;
                yPosInCu = (idx/cuWidth);
                xPosInCtu = ALL_X_POS[cuSizeIdx][currCu]; // (currCu%cuColumnsPerCtu)*cuWidth + xPosInCu;
                yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu]; // (currCu/cuColumnsPerCtu)*cuHeight + yPosInCu;

                isMiddle = yPosInCu>=upsamplingVertical; // In this case, the top boundary is not used
                offsetInStride = yPosInCu%upsamplingVertical+1; // Position inside one window where samples are being interpolated. BeforeReference has stride=0, first interpolated sample has stride=1            }

                // For the first couple of sample columns, the "before" reference is the top boundaries buffer
                if(isMiddle == 0){
                    // Predicted value that is before the current sample
                    valueBefore = refT[cuIdxInIteration*32 + xPosInCu];
                    // Predicted value that is after the current sample
                    valueAfter = localUpsampledPrediction[cuIdxInIteration][(((yPosInCu>>log2UpsamplingVertical)<<log2UpsamplingVertical) + upsamplingVertical-1)*cuWidth + xPosInCu];
                }
                else{ // isMiddle == 1
                    valueBefore = localUpsampledPrediction[cuIdxInIteration][(((yPosInCu>>log2UpsamplingVertical)<<log2UpsamplingVertical)-1)*cuWidth + xPosInCu];
                    valueAfter  = localUpsampledPrediction[cuIdxInIteration][(((yPosInCu>>log2UpsamplingVertical)<<log2UpsamplingVertical)+upsamplingVertical-1)*cuWidth + xPosInCu];
                }

                int filteredSample = ((upsamplingVertical-offsetInStride)*valueBefore + offsetInStride*valueAfter + roundingOffsetVertical)>>log2UpsamplingVertical;
                localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth+xPosInCu] = filteredSample;                
            }

            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish the vertical upsampling

            /* TRACE COMPLETE UPSAMPLING
            // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && currCu==3 && mode==0 && lid%itemsPerCuInUpsampling==0){
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && mode==0 && lid%itemsPerCuInUpsampling==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && mode==0 && lid%itemsPerCuInUpsampling==0){
                printf("UPSAMPLED PREDICTION,CTU=%d,WH=%dx%d,CU=%d\n", ctuIdx, cuWidth, cuHeight, currCu);
                for(int i=0; i<cuHeight; i++){
                    for(int j=0; j<cuWidth; j++){
                        printf("%d,", localUpsampledPrediction[cuIdxInIteration][i*cuWidth+j]);
                    }
                    printf("\n");
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/

            
            // At this point the upsampling for the current mode is complete. We can compute the distortion...
            
            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      COMPUTE SAD FOR THE CURRENT CU

            // Here, each workitem will compute the SAD at one or more position, and accumulate the result in localSAD[lid]
            // At the end we must reduce the valeus between 0-127 and 128-255 to obtain the final SAD of each CU
            localSAD[lid] = 0;
            localSATD[lid] = 0;

            int nPassesForSad = nPassesOriginalFetch;

            int NEW_SATD = 0;
            int CONDUCT_SATD = 1;

            for(int pass=0; pass<nPassesForSad; pass++){
                idx = pass*itemsPerCuInFetchOriginal + lid%itemsPerCuInFetchOriginal;
                xPosInCu = idx%cuWidth;
                yPosInCu = idx/cuWidth;
                
                localSAD[lid] += (int) abs(localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] - localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu]);
                // Substitute predicted samples by prediction error
                if(NEW_SATD)
                    localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] = localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] - localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu];
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish their SAD computation
            

            // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && currCu==3 && lid%128==0 && mode==0){
            //     printf("SERIAL SAD FOR,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
            //     for(int i=0; i<128; i++){
            //         printf("%d,", localSAD[lid+i]);
            //     }
            //     printf("\n");
            // }

            // PARALLEL REDUCTION
            int nPassesParallelSum = 5; // log2(32)
            int stride = 16;
            int baseIdx = (lid/32)*32 + lid%32;
            for(int pass=0; pass<nPassesParallelSum; pass++){
                if(lid%32 < stride){
                    localSAD[baseIdx] = localSAD[baseIdx] + localSAD[baseIdx+stride];
                    stride = stride/2;
                }    
                barrier(CLK_LOCAL_MEM_FENCE);  
            }

            /* TRACE SAD
            if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && currCu==0 && lid%128==0){
                printf("SAD,CTU=%d,WH=%dx%d,CU=%d,MODE=%d,SAD=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode, localSAD[(cuIdxInIteration%2)*128]);
                // printf("OI,localSAD[lid]=%d\n", localSAD[lid]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/      
            
            localSATD[lid] = 0;

            // ALLOW SKIPPING SATD TO MEASURE PROCESSING TIME
            if(CONDUCT_SATD){
                // TODO: This NEW_SATD method WAS NOT improved to support the remaining block sizes and alignments
                if(NEW_SATD){
                    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                    //
                    //      ALTERNATIVE SATD COMPUTATION
                    int subblocksPerCu = (cuWidth*cuHeight)/16;
                    int subblocksColumnsPerCu = cuWidth/4;
                    int nPasses = (cuWidth*cuHeight)/32;
                    
                    // TODO: Transform this section into a function
                    // Compute transformed differences at the 4x4-level
                    for(int pass=0; pass<nPasses; pass++){
                        int idx = pass*32 + lid%32;
                        int currSubblock = idx/16;
                        int subblockX = (currSubblock%subblocksColumnsPerCu)*4;
                        int subblockY = (currSubblock/subblocksColumnsPerCu)*4;
                        // Position of the sample that will be computed
                        int i = idx%16;
                        int i_xInCu = subblockX + i%4;
                        int i_yInCu = subblockY + i/4;
                        int i_idxInCu = i_yInCu*cuWidth + i_xInCu;
                        // Used to index the arguments of the SATD for one position, and store the temporary value before updating the array
                        int j, j_xInCu, j_yInCu, k, k_xInCu, k_yInCu, f_i;
                        int j_idxInCu, k_idxInCu;
                        int tempValue;

                        // Now the hadamard transform is applied to the 4x4 blocks. We derive the index of operands and operation (+/-) based on the index of current coefficient
                        // m[i] = diff[j] +/-(f_i) diff[k]
                        // The values of j, f_i and k are computed based on i

                        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                        // 
                        // 1st
                        j = i - 4*(i>=8) - 8*(i>=12);
                        k = 12 + i - 8*(i>=4) - 4*(i>=8);
                        f_i = i<8 ? 1 : -1;
                        
                        j_xInCu = subblockX + j%4;
                        j_yInCu = subblockY + j/4;
                        j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                        k_xInCu = subblockX + k%4;
                        k_yInCu = subblockY + k/4;
                        k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                        // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                        tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                        localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                        // 
                        // 2nd
                        j = i + 4*(i>=4) - 12*(i>=8) + 8*(i>=12);
                        k = 4 + i + 4*(i>=4) - 12*(i>=8);
                        f_i = i<8 ? 1 : -1;
                        
                        j_xInCu = subblockX + j%4;
                        j_yInCu = subblockY + j/4;
                        j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                        k_xInCu = subblockX + k%4;
                        k_yInCu = subblockY + k/4;
                        k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                        // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                        tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                        localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                        // barrier(CLK_LOCAL_MEM_FENCE);                


                        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                        // 
                        // 3rd
                        j = 4*(i/4) + 1*(i%4==1) + 1*(i%4==2);
                        k = 3 + 4*(i/4) - 1*(i%4==1) - 1*(i%4==2);
                        f_i = i%4<2 ? 1 : -1;
                        
                        j_xInCu = subblockX + j%4;
                        j_yInCu = subblockY + j/4;
                        j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                        k_xInCu = subblockX + k%4;
                        k_yInCu = subblockY + k/4;
                        k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                        // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                        tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                        localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                        // barrier(CLK_LOCAL_MEM_FENCE);  

                        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                        // 
                        // 4th
                        j = 4*(i/4) + 2*(i%4==2) + 3*(i%4==3);
                        k = 1 + 4*(i/4) + 2*(i%4==2) + 1*(i%4==3);
                        f_i = i%2==0 ? 1 : -1;
                        
                        j_xInCu = subblockX + j%4;
                        j_yInCu = subblockY + j/4;
                        j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                        k_xInCu = subblockX + k%4;
                        k_yInCu = subblockY + k/4;
                        k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                        // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                        tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                        // barrier(CLK_LOCAL_MEM_FENCE);

                        // TODO: MAYBE it is not necessary to synch the workitems here because the next iteration will work over a different set of values. Only synch outside the for loop
                        // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                        localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                        barrier(CLK_LOCAL_MEM_FENCE);  
                    }


                    //* 
                    // REDUCE TOTAL SATD IN PARALLEL
                    // At this point we have the transformed differences for all subblocks 4x4.
                    // Now each workitem will sum the absolute coefficients and save in its position in localSATD[lid]
                    // Then, these values are reduced
                    itemsPerCuInSatd = min(128, (cuWidth*cuHeight)/16);
                    int nPassesSumCoefficients = max(1, subblocksPerCu/itemsPerCuInSatd);
                    if(lid%128<itemsPerCuInSatd){
                        int sb, subblockX, subblockY, startRow, currSATD;
                        for(int pass=0; pass<nPassesSumCoefficients; pass++){
                            sb = pass*itemsPerCuInSatd + lid%128; // Subblock this workitem will process
                            currSATD = 0;
                            subblockX = (sb%subblocksColumnsPerCu)*4;
                            subblockY = (sb/subblocksColumnsPerCu)*4;
                            startRow = subblockY*cuWidth + subblockX;
                            for(int row=0; row<4; row++){
                                for(int col=0; col<4; col++){
                                    currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow + row*cuWidth + col]);
                                }
                            }
                            currSATD -= abs(localUpsampledPrediction[cuIdxInIteration][startRow]);
                            currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow]) >> 2;
                            currSATD = ((currSATD+1)>>1);
                            
                            // if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && currCu==0 && mode==0){// } && lid%128==0){
                            //     printf("CTU=%d,cuSize=%d,cuIdx=%d,sb=%d,SATD=%d\n", ctuIdx, cuSizeIdx, currCu, sb, currSATD);
                            // }


                            localSATD[lid] += currSATD;
                            currSATD = 0;
                        }
                    }
                    barrier(CLK_LOCAL_MEM_FENCE);  // Wait until all partial SATDs are computed
                    //*/

                    //*
                    // PARALLEL REDUCTION
                    nPassesParallelSum = (int) log2( (float) min(128, cuWidth*cuHeight/16) ); // log2()
                    stride = itemsPerCuInSatd/2;
                    baseIdx = (lid/128)*128 + lid%128;
                    for(int pass=0; pass<nPassesParallelSum; pass++){
                        if(lid%128 < stride){
                            localSATD[baseIdx] = localSATD[baseIdx] + localSATD[baseIdx+stride];
                            stride = stride/2;
                        }    
                        barrier(CLK_LOCAL_MEM_FENCE);  
                    }
                    //*/
                    
                    
                    /*
                    // Sum individual values and reduce SATD in sequential fashion
                    if(lid%128 == 0){
                        int currSATD = 0, totalSATD = 0;
                        for(int sb=1; sb<subblocksPerCu; sb++){
                            localSATD[lid] += localSATD[lid+sb];
                            // int subblockX = (sb%subblocksColumnsPerCu)*4;
                            // int subblockY = (sb/subblocksColumnsPerCu)*4;
                            // int startRow = subblockY*cuWidth + subblockX;
                            // for(int row=0; row<4; row++){
                            //     for(int col=0; col<4; col++)
                            //         currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow + row*cuWidth + col]);
                            // }
                            // currSATD -= abs(localUpsampledPrediction[cuIdxInIteration][startRow]);
                            // currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow]) >> 2;
                            // currSATD = (currSATD+1)>>1;
                            
                            // totalSATD += currSATD;

                            // // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && mode==11 && currCu==2 && lid%128==0){
                            // //     // printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %ld, Total SATD %ld\n", lid, idx, subblockX, subblockY, currSATD, totalSATD);
                            // //     printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %d, Total SATD %d\n", lid, idx, subblockX, subblockY, currSATD, totalSATD);
                            // // }
                            // currSATD = 0;
                        }
                        // localSATD[lid] = totalSATD;
                    }
                    //*/
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                
                
                if(!NEW_SATD){
                    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                    //
                    //      COMPUTE SATD FOR THE CURRENT CU
                    //*
                    int idxSubblock;
                    int nPassesForSatd = max(1,nPassesOriginalFetch/16);
                    itemsPerCuInSatd = min(32, (cuWidth*cuHeight)/16);
                    int subblockX, subblockY;
                    short16 origSubblock, predSubblock;

                    // Here, each workitem will compute the SATD for one or more subblocks 4x4, and accumulate the results in __local localSATD
                    if((lid%32) < itemsPerCuInSatd){
                        for(int pass=0; pass<nPassesForSatd; pass++){
                            idxSubblock = pass*itemsPerCuInSatd + lid%32;
                            subblockX = (idxSubblock%(cuWidth/4))<<2;
                            subblockY = (idxSubblock/(cuWidth/4))<<2;
                            idx = subblockY*(cuWidth/4) + subblockX/4;


                            // 1st row                    
                            origSubblock.lo.lo = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.lo.lo = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                            // 2nd row
                            idx += cuWidth/4;
                            origSubblock.lo.hi = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.lo.hi = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                            // 3rd row
                            idx += cuWidth/4;
                            origSubblock.hi.lo = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.hi.lo = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                            // 4th row
                            idx += cuWidth/4;
                            origSubblock.hi.hi = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.hi.hi = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);

                            localSATD[lid] += satd_4x4(origSubblock, predSubblock);
                            
                            /* TRACE INTERMEDIARY SATD 4X4
                            if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && mode==0 && currCu==0){//} && lid==0){
                                printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %ld\n", lid, idx, subblockX, subblockY, satd_4x4(origSubblock, predSubblock));
                                // printf("Orig subblock\n");
                                // printf("%d,%d,%d,%d\n", origSubblock.s0, origSubblock.s1, origSubblock.s2, origSubblock.s3);
                                // printf("%d,%d,%d,%d\n", origSubblock.s4, origSubblock.s5, origSubblock.s6, origSubblock.s7);
                                // printf("%d,%d,%d,%d\n", origSubblock.s8, origSubblock.s9, origSubblock.sa, origSubblock.sb);
                                // printf("%d,%d,%d,%d\n", origSubblock.sc, origSubblock.sd, origSubblock.se, origSubblock.sf);
                            
                                // printf("Pred subblock\n");
                                // printf("%d,%d,%d,%d\n", predSubblock.s0, predSubblock.s1, predSubblock.s2, predSubblock.s3);
                                // printf("%d,%d,%d,%d\n", predSubblock.s4, predSubblock.s5, predSubblock.s6, predSubblock.s7);
                                // printf("%d,%d,%d,%d\n", predSubblock.s8, predSubblock.s9, predSubblock.sa, predSubblock.sb);
                                // printf("%d,%d,%d,%d\n\n\n", predSubblock.sc, predSubblock.sd, predSubblock.se, predSubblock.sf);                    
                            }
                            //*/
                            
                            //*
                        }
                    }

                    barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish their SATD computation

                    // PARALLEL REDUCTION
                    nPassesParallelSum = (int) log2( (float) min(32, cuWidth*cuHeight/16) ); // log2()
                    stride = itemsPerCuInSatd/2;
                    baseIdx = (lid/32)*32 + lid%32;
                    for(int pass=0; pass<nPassesParallelSum; pass++){
                        if(lid%32 < stride){
                            localSATD[baseIdx] = localSATD[baseIdx] + localSATD[baseIdx+stride];
                            stride = stride/2;
                        }    
                        barrier(CLK_LOCAL_MEM_FENCE);  
                    }
                }
            }


            //*/

            // Save SAD and SATD of current CU/mode in a __local buffer. We only access global memory when all SAD values are computed or all CUs
            if(lid==0){
                localSadEntireCtu[firstCu][mode] = localSAD[0];
                localSadEntireCtu[firstCu+1][mode] = localSAD[32];
                localSadEntireCtu[firstCu+2][mode] = localSAD[64];
                localSadEntireCtu[firstCu+3][mode] = localSAD[96];
                localSadEntireCtu[firstCu+4][mode] = localSAD[128];
                localSadEntireCtu[firstCu+5][mode] = localSAD[160];
                localSadEntireCtu[firstCu+6][mode] = localSAD[192];
                localSadEntireCtu[firstCu+7][mode] = localSAD[224];

                localSatdEntireCtu[firstCu][mode] = localSATD[0];
                localSatdEntireCtu[firstCu+1][mode] = localSATD[32];
                localSatdEntireCtu[firstCu+2][mode] = localSATD[64];
                localSatdEntireCtu[firstCu+3][mode] = localSATD[96];
                localSatdEntireCtu[firstCu+4][mode] = localSATD[128];
                localSatdEntireCtu[firstCu+5][mode] = localSATD[160];
                localSatdEntireCtu[firstCu+6][mode] = localSATD[192];
                localSatdEntireCtu[firstCu+7][mode] = localSATD[224];
            } 

        } // Finish current mode
    } // Finish current pair of CUs   
    
    // When all CUs are processed, we move the results into global buffer
    if(lid < ALL_cusPerCtu[cuSizeIdx]*numPredictionModes*2){
        int nPassesMoveSadIntoGlobal = max(1, (int)ceil((1.0*ALL_cusPerCtu[cuSizeIdx]*numPredictionModes*2)/wgSize));
        int idxInLocal, cu, mode, idxInGlobal;
        
        for(int pass=0; pass<nPassesMoveSadIntoGlobal; pass++){
            idxInLocal = pass*wgSize + lid;
            cu = idxInLocal / (numPredictionModes*2);
            mode = idxInLocal % (numPredictionModes*2);

            // Point to current CTU in global buffer
            idxInGlobal = ctuIdx*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES];
            // Point to current CU size in global buffer
            idxInGlobal    += ALL_stridedDistortionsPerCtu[cuSizeIdx];
            // Point to current CU and mode in global buffer
            idxInGlobal    += cu*numPredictionModes*2 + mode;

            if(cu < ALL_cusPerCtu[cuSizeIdx]){
                SAD[idxInGlobal] = ( long ) localSadEntireCtu[cu][mode];
                SATD[idxInGlobal] = ( long ) localSatdEntireCtu[cu][mode];
                minSadHad[idxInGlobal] = (long) min(2*localSadEntireCtu[cu][mode], localSatdEntireCtu[cu][mode]);
            }
        }
    }
}

__kernel void upsampleDistortionSizeId0_ALL(__global short *reducedPrediction, const int frameWidth, const int frameHeight, __global long *SAD, __global long *SATD, __global short* originalSamples, __global short *unified_redT, __global short *unified_redL, __global short *unified_refT, __global short *unified_refL, __global long *minSadHad){
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    const short ctuIdx = wg/8; // 8 because each wg processes only 12.5% of the CTU
    const short cuSizeIdx = ALL_AL_4x4; //wg%NUM_CU_SIZES_SizeId0 + FIRST_SizeId0;
    const short cuWidth = ALL_widths[cuSizeIdx];
    const short cuHeight = ALL_heights[cuSizeIdx];
    const int nCusInCtu = 128; // Although there are 1024 CUs 4x4, each WG process only 128 (8 WGs together process one CTU)

    const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
      
    const short cuColumnsPerCtu = ALL_cuColumnsPerCtu[cuSizeIdx];
    const short cuRowsPerCtu = ALL_cuRowsPerCtu[cuSizeIdx]/8;

    // CTU position inside the frame
    const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
    const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

    const int numPredictionModes = PREDICTION_MODES_ID0;
    const char reducedBoundarySize = BOUNDARY_SIZE_Id0;
    const char reducedPredSize = REDUCED_PRED_SIZE_Id0;

    const int upsamplingHorizontal = cuWidth / reducedPredSize;
    const int upsamplingVertical = cuHeight / reducedPredSize;

    const int log2UpsamplingHorizontal = (int) log2((float) upsamplingHorizontal);
    const int roundingOffsetHorizontal = 1 << (log2UpsamplingHorizontal - 1);

    const int log2UpsamplingVertical = (int) log2((float) upsamplingVertical);
    const int roundingOffsetVertical = 1 << (log2UpsamplingVertical - 1);
    
    // TODO: Correct this when supporting more block sizes.
    // Correct value is upsamplingHorizontal>1 || upsamplingVertical>1;
    int needUpsampling = 0; 

    // ######################################################################
    //      Variables shared for horizontal and vertical interpolation
    int xPosInCu, yPosInCu, xPosInCtu, yPosInCtu, xPosInFrame, yPosInFrame, idx;
    int valueBefore, valueAfter, beforeIdx, afterIdx;
    int isMiddle;
    int offsetInStride;
    int itemsPerCuInUpsampling;
    int itemsPerCuInFetchOriginal;
    int itemsPerCuInSatd;
    
    // During upsampling, 16 workitems are assigned to conduct the processing of each CU (i.e., with wgSize=256 we process 16 CUs at once) This is based on the size of the smaller CU 4x4=16
    // We fetch the boundaries of CUs depending on wgSize, upsample these CUs with all prediction modes to reuse the boundaries without extra memory access
    // Compute the distortion for these CUs with each prediction mode, then process the next CUs
    __local short localReducedPrediction[16][REDUCED_PRED_SIZE_Id0*REDUCED_PRED_SIZE_Id0]; // At most, 16 CUs are processed simultaneously
    // ReducedPrediction is equal to upsampled for 4x4 CUs, since there is no upsampling
    // __local short localUpsampledPrediction[8][32*4];
    __local short localOriginalSamples[16][4*4];
    __local int localSAD[256], localSATD[256];

    __local int localSadEntireCtu[128][PREDICTION_MODES_ID0*2];
    __local int localSatdEntireCtu[128][PREDICTION_MODES_ID0*2];

    // Each CU will be upsampled using 16 workitems, irrespective of the CU size
    // We will process 16 CUs simultaneously when wgSize=256
    itemsPerCuInUpsampling = 16;
    itemsPerCuInFetchOriginal = 16;

    int cuIdxOffset = 128 * (wg%8); // Each WG processes 12.5% of the CTU, therefore we must correct the first sample where each WG start
    int offsetInCtuY = 4*(cuIdxOffset/cuColumnsPerCtu);

    for(int firstCu = 0; firstCu < nCusInCtu; firstCu += wgSize/itemsPerCuInUpsampling){
        int cuIdxInIteration = lid/itemsPerCuInUpsampling; // This represents if the current CU equals firstCU, firstCU+1, firstCU+2, firstCU+3, ...
        int currCu = cuIdxOffset + firstCu + cuIdxInIteration;

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //      FETCH THE  ORIGINAL SAMPLES FOR THE CUs BEING PROCESSED
        
        int nPassesOriginalFetch = (cuWidth*cuHeight)/itemsPerCuInFetchOriginal;
        int xPosInCu, yPosInCu, xPosInCtu, yPosInCtu, xPosInFrame, yPosInFrame;

        for(int pass=0; pass<nPassesOriginalFetch; pass++){
            idx = pass*itemsPerCuInFetchOriginal + lid%itemsPerCuInFetchOriginal;
            xPosInCu = idx%cuWidth;
            yPosInCu = idx/cuWidth;
            xPosInCtu = 4*(currCu%cuColumnsPerCtu) + xPosInCu;
            yPosInCtu =  4*(currCu/cuColumnsPerCtu) + yPosInCu;

            xPosInFrame = ctuX + xPosInCtu;
            yPosInFrame = ctuY + yPosInCtu;
            
            localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] = originalSamples[yPosInFrame*frameWidth + xPosInFrame];

            /*
            barrier(CLK_LOCAL_MEM_FENCE);
            if(ctuIdx==16 && currCu==0 && lid%16==0){
                printf("ORIG SAMPLES CU %d\n", currCu);
                for(int i=0; i<4; i++){
                    for(int j=0; j<4; j++){
                        printf("%d,", localOriginalSamples[cuIdxInIteration][i*cuWidth + j]);
                    }
                    printf("\n");
                }
            }
            //*/
        }
        

        // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        //
        //      FETCH THE BOUNDARIES REQUIRED FOR UPSAMPLING
        
        // No need to fetch boundaries because there is no upsampling

        int idxCurrCuAndMode = 0;
        // Now we do the upsampling for all prediction modes of the current 16 CUs
        for(int mode=0; mode<numPredictionModes*2; mode++){
            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      FETCH THE REDUCED PREDICTION FOR CURRENT MODE AND CUs
    
            // Point to the start of current CTU
            idxCurrCuAndMode = ctuIdx*ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES];
            // Point to start of current CU size
            idxCurrCuAndMode += ALL_stridedPredictionsPerCtu[cuSizeIdx];
            // Point to start of current CU
            idxCurrCuAndMode += currCu*reducedPredSize*reducedPredSize*numPredictionModes*2;
            // Point to start of current prediction mode
            idxCurrCuAndMode += mode*reducedPredSize*reducedPredSize;

            if(lid%itemsPerCuInUpsampling < (reducedPredSize*reducedPredSize)){
                localReducedPrediction[cuIdxInIteration][lid%itemsPerCuInUpsampling] = reducedPrediction[idxCurrCuAndMode + lid%itemsPerCuInUpsampling];
            }

            barrier(CLK_LOCAL_MEM_FENCE); // Wait until the whole buffer for reduced prediction is filled


            /* Trace reduced prediction fetched from global memory
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_AL_4x4 && currCu==0 && mode==0 && lid%itemsPerCuInUpsampling==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && mode==0 && lid%itemsPerCuInUpsampling==0){
                printf("REDUCED PREDICTION,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
                for(int i=0; i<4; i++){
                    for(int j=0; j<4; j++){
                        printf("%d,", localReducedPrediction[cuIdxInIteration][i*4+j]);
                    }
                    printf("\n");
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/


            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      START WITH HORIZONTAL UPSAMPLING...

            //      NO NEED TO UPSAMPLE BECAUSE REDUCED PREDICTION AND ORIGINAL BLOCK HAVE SAME DIMENSIONS 4x4

            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      PROCEED WITH VERTICAL UPSAMPLING...

            //      NO NEED TO UPSAMPLE BECAUSE REDUCED PREDICTION AND ORIGINAL BLOCK HAVE SAME DIMENSIONS 4x4

            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      COMPUTE SAD FOR THE CURRENT CU

            // Here, each workitem will compute the SAD at one or more position, and accumulate the result in localSAD[lid]
            // At the end we must reduce the valeus between 0-127 and 128-255 to obtain the final SAD of each CU
            localSAD[lid] = 0;
            localSATD[lid] = 0;

            int nPassesForSad = nPassesOriginalFetch;

            int NEW_SATD = 0;
            int CONDUCT_SATD = 1;

            for(int pass=0; pass<nPassesForSad; pass++){
                idx = pass*itemsPerCuInFetchOriginal + lid%itemsPerCuInFetchOriginal;
                xPosInCu = idx%cuWidth;
                yPosInCu = idx/cuWidth;
                
                localSAD[lid] += (int) abs(localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] - localReducedPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu]);
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish their SAD computation
            

            // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && currCu==3 && lid%128==0 && mode==0){
            //     printf("SERIAL SAD FOR,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
            //     for(int i=0; i<128; i++){
            //         printf("%d,", localSAD[lid+i]);
            //     }
            //     printf("\n");
            // }

            // PARALLEL REDUCTION
            int nPassesParallelSum = 4; // log2(16)
            int stride = 8;
            int baseIdx = (lid/16)*16 + lid%16;
            for(int pass=0; pass<nPassesParallelSum; pass++){
                if(lid%16 < stride){
                    localSAD[baseIdx] = localSAD[baseIdx] + localSAD[baseIdx+stride];
                    stride = stride/2;
                }    
                barrier(CLK_LOCAL_MEM_FENCE);  
            }

            /* TRACE SAD
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_AL_4x4 && currCu==0 && lid%16==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_AL_4x4 && currCu==0 && lid%16==0){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_AL_4x4 && currCu==1023){                
                printf("SAD,CTU=%d,WH=%dx%d,CU=%d,MODE=%d,SAD=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode, localSAD[0]);
                // printf("OI,localSAD[lid]=%d\n", localSAD[lid]);
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            //*/      
            
            localSATD[lid] = 0;

            // ALLOW SKIPPING SATD TO MEASURE PROCESSING TIME
            if(CONDUCT_SATD){
                // TODO: This NEW_SATD method WAS NOT improved to support the remaining block sizes and alignments
                // if(NEW_SATD){
                //     // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                //     //
                //     //      ALTERNATIVE SATD COMPUTATION
                //     int subblocksPerCu = (cuWidth*cuHeight)/16;
                //     int subblocksColumnsPerCu = cuWidth/4;
                //     int nPasses = (cuWidth*cuHeight)/32;
                    
                //     // TODO: Transform this section into a function
                //     // Compute transformed differences at the 4x4-level
                //     for(int pass=0; pass<nPasses; pass++){
                //         int idx = pass*32 + lid%32;
                //         int currSubblock = idx/16;
                //         int subblockX = (currSubblock%subblocksColumnsPerCu)*4;
                //         int subblockY = (currSubblock/subblocksColumnsPerCu)*4;
                //         // Position of the sample that will be computed
                //         int i = idx%16;
                //         int i_xInCu = subblockX + i%4;
                //         int i_yInCu = subblockY + i/4;
                //         int i_idxInCu = i_yInCu*cuWidth + i_xInCu;
                //         // Used to index the arguments of the SATD for one position, and store the temporary value before updating the array
                //         int j, j_xInCu, j_yInCu, k, k_xInCu, k_yInCu, f_i;
                //         int j_idxInCu, k_idxInCu;
                //         int tempValue;

                //         // Now the hadamard transform is applied to the 4x4 blocks. We derive the index of operands and operation (+/-) based on the index of current coefficient
                //         // m[i] = diff[j] +/-(f_i) diff[k]
                //         // The values of j, f_i and k are computed based on i

                //         // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                //         // 
                //         // 1st
                //         j = i - 4*(i>=8) - 8*(i>=12);
                //         k = 12 + i - 8*(i>=4) - 4*(i>=8);
                //         f_i = i<8 ? 1 : -1;
                        
                //         j_xInCu = subblockX + j%4;
                //         j_yInCu = subblockY + j/4;
                //         j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                //         k_xInCu = subblockX + k%4;
                //         k_yInCu = subblockY + k/4;
                //         k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                //         // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                //         tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                //         // barrier(CLK_LOCAL_MEM_FENCE);

                //         // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                //         localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                //         // barrier(CLK_LOCAL_MEM_FENCE);

                //         // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                //         // 
                //         // 2nd
                //         j = i + 4*(i>=4) - 12*(i>=8) + 8*(i>=12);
                //         k = 4 + i + 4*(i>=4) - 12*(i>=8);
                //         f_i = i<8 ? 1 : -1;
                        
                //         j_xInCu = subblockX + j%4;
                //         j_yInCu = subblockY + j/4;
                //         j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                //         k_xInCu = subblockX + k%4;
                //         k_yInCu = subblockY + k/4;
                //         k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                //         // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                //         tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                //         // barrier(CLK_LOCAL_MEM_FENCE);

                //         // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                //         localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                //         // barrier(CLK_LOCAL_MEM_FENCE);                


                //         // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                //         // 
                //         // 3rd
                //         j = 4*(i/4) + 1*(i%4==1) + 1*(i%4==2);
                //         k = 3 + 4*(i/4) - 1*(i%4==1) - 1*(i%4==2);
                //         f_i = i%4<2 ? 1 : -1;
                        
                //         j_xInCu = subblockX + j%4;
                //         j_yInCu = subblockY + j/4;
                //         j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                //         k_xInCu = subblockX + k%4;
                //         k_yInCu = subblockY + k/4;
                //         k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                //         // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                //         tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                //         // barrier(CLK_LOCAL_MEM_FENCE);

                //         // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                //         localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                //         // barrier(CLK_LOCAL_MEM_FENCE);  

                //         // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                //         // 
                //         // 4th
                //         j = 4*(i/4) + 2*(i%4==2) + 3*(i%4==3);
                //         k = 1 + 4*(i/4) + 2*(i%4==2) + 1*(i%4==3);
                //         f_i = i%2==0 ? 1 : -1;
                        
                //         j_xInCu = subblockX + j%4;
                //         j_yInCu = subblockY + j/4;
                //         j_idxInCu = j_yInCu*cuWidth + j_xInCu;

                //         k_xInCu = subblockX + k%4;
                //         k_yInCu = subblockY + k/4;
                //         k_idxInCu = k_yInCu*cuWidth + k_xInCu;
                //         // compute m[i] but do not overwrite the original buffer yet. Wait until all items have computed their temporary values
                //         tempValue = localUpsampledPrediction[cuIdxInIteration][j_idxInCu] + f_i*localUpsampledPrediction[cuIdxInIteration][k_idxInCu];
                //         // barrier(CLK_LOCAL_MEM_FENCE);

                //         // TODO: MAYBE it is not necessary to synch the workitems here because the next iteration will work over a different set of values. Only synch outside the for loop
                //         // Update the matrix with the updated value and synch between workitems. These values will be used to compute the next temp value
                //         localUpsampledPrediction[cuIdxInIteration][i_idxInCu] = tempValue;
                //         barrier(CLK_LOCAL_MEM_FENCE);  
                //     }


                //     //* 
                //     // REDUCE TOTAL SATD IN PARALLEL
                //     // At this point we have the transformed differences for all subblocks 4x4.
                //     // Now each workitem will sum the absolute coefficients and save in its position in localSATD[lid]
                //     // Then, these values are reduced
                //     itemsPerCuInSatd = min(128, (cuWidth*cuHeight)/16);
                //     int nPassesSumCoefficients = max(1, subblocksPerCu/itemsPerCuInSatd);
                //     if(lid%128<itemsPerCuInSatd){
                //         int sb, subblockX, subblockY, startRow, currSATD;
                //         for(int pass=0; pass<nPassesSumCoefficients; pass++){
                //             sb = pass*itemsPerCuInSatd + lid%128; // Subblock this workitem will process
                //             currSATD = 0;
                //             subblockX = (sb%subblocksColumnsPerCu)*4;
                //             subblockY = (sb/subblocksColumnsPerCu)*4;
                //             startRow = subblockY*cuWidth + subblockX;
                //             for(int row=0; row<4; row++){
                //                 for(int col=0; col<4; col++){
                //                     currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow + row*cuWidth + col]);
                //                 }
                //             }
                //             currSATD -= abs(localUpsampledPrediction[cuIdxInIteration][startRow]);
                //             currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow]) >> 2;
                //             currSATD = ((currSATD+1)>>1);
                            
                //             // if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && currCu==0 && mode==0){// } && lid%128==0){
                //             //     printf("CTU=%d,cuSize=%d,cuIdx=%d,sb=%d,SATD=%d\n", ctuIdx, cuSizeIdx, currCu, sb, currSATD);
                //             // }


                //             localSATD[lid] += currSATD;
                //             currSATD = 0;
                //         }
                //     }
                //     barrier(CLK_LOCAL_MEM_FENCE);  // Wait until all partial SATDs are computed
                //     //*/

                //     //*
                //     // PARALLEL REDUCTION
                //     nPassesParallelSum = (int) log2( (float) min(128, cuWidth*cuHeight/16) ); // log2()
                //     stride = itemsPerCuInSatd/2;
                //     baseIdx = (lid/128)*128 + lid%128;
                //     for(int pass=0; pass<nPassesParallelSum; pass++){
                //         if(lid%128 < stride){
                //             localSATD[baseIdx] = localSATD[baseIdx] + localSATD[baseIdx+stride];
                //             stride = stride/2;
                //         }    
                //         barrier(CLK_LOCAL_MEM_FENCE);  
                //     }
                //     //*/
                    
                    
                //     /*
                //     // Sum individual values and reduce SATD in sequential fashion
                //     if(lid%128 == 0){
                //         int currSATD = 0, totalSATD = 0;
                //         for(int sb=1; sb<subblocksPerCu; sb++){
                //             localSATD[lid] += localSATD[lid+sb];
                //             // int subblockX = (sb%subblocksColumnsPerCu)*4;
                //             // int subblockY = (sb/subblocksColumnsPerCu)*4;
                //             // int startRow = subblockY*cuWidth + subblockX;
                //             // for(int row=0; row<4; row++){
                //             //     for(int col=0; col<4; col++)
                //             //         currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow + row*cuWidth + col]);
                //             // }
                //             // currSATD -= abs(localUpsampledPrediction[cuIdxInIteration][startRow]);
                //             // currSATD += abs(localUpsampledPrediction[cuIdxInIteration][startRow]) >> 2;
                //             // currSATD = (currSATD+1)>>1;
                            
                //             // totalSATD += currSATD;

                //             // // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && mode==11 && currCu==2 && lid%128==0){
                //             // //     // printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %ld, Total SATD %ld\n", lid, idx, subblockX, subblockY, currSATD, totalSATD);
                //             // //     printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %d, Total SATD %d\n", lid, idx, subblockX, subblockY, currSATD, totalSATD);
                //             // // }
                //             // currSATD = 0;
                //         }
                //         // localSATD[lid] = totalSATD;
                //     }
                //     //*/
                //     barrier(CLK_LOCAL_MEM_FENCE);
                // }
                
                
                if(!NEW_SATD){
                    // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                    //
                    //      COMPUTE SATD FOR THE CURRENT CU
                    //*
                    int idxSubblock;
                    int nPassesForSatd = max(1,nPassesOriginalFetch/16);
                    itemsPerCuInSatd = min(16, (cuWidth*cuHeight)/16);
                    int subblockX, subblockY;
                    short16 origSubblock, predSubblock;

                    // Here, each workitem will compute the SATD for one or more subblocks 4x4, and accumulate the results in __local localSATD
                    if((lid%16) < itemsPerCuInSatd){
                        for(int pass=0; pass<nPassesForSatd; pass++){
                            idxSubblock = pass*itemsPerCuInSatd + lid%16;
                            subblockX = (idxSubblock%(cuWidth/4))<<2;
                            subblockY = (idxSubblock/(cuWidth/4))<<2;
                            idx = subblockY*(cuWidth/4) + subblockX/4;


                            // 1st row                    
                            origSubblock.lo.lo = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.lo.lo = vload4(idx, localReducedPrediction[cuIdxInIteration]);
                            // 2nd row
                            idx += cuWidth/4;
                            origSubblock.lo.hi = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.lo.hi = vload4(idx, localReducedPrediction[cuIdxInIteration]);
                            // 3rd row
                            idx += cuWidth/4;
                            origSubblock.hi.lo = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.hi.lo = vload4(idx, localReducedPrediction[cuIdxInIteration]);
                            // 4th row
                            idx += cuWidth/4;
                            origSubblock.hi.hi = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                            predSubblock.hi.hi = vload4(idx, localReducedPrediction[cuIdxInIteration]);

                            localSATD[lid] += satd_4x4(origSubblock, predSubblock);
                            
                            /* TRACE INTERMEDIARY SATD 4X4
                            if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && mode==0 && currCu==0){//} && lid==0){
                                printf("lid=%d, idx=%d, subX=%d, subY=%d, Partial SATD %ld\n", lid, idx, subblockX, subblockY, satd_4x4(origSubblock, predSubblock));
                                // printf("Orig subblock\n");
                                // printf("%d,%d,%d,%d\n", origSubblock.s0, origSubblock.s1, origSubblock.s2, origSubblock.s3);
                                // printf("%d,%d,%d,%d\n", origSubblock.s4, origSubblock.s5, origSubblock.s6, origSubblock.s7);
                                // printf("%d,%d,%d,%d\n", origSubblock.s8, origSubblock.s9, origSubblock.sa, origSubblock.sb);
                                // printf("%d,%d,%d,%d\n", origSubblock.sc, origSubblock.sd, origSubblock.se, origSubblock.sf);
                            
                                // printf("Pred subblock\n");
                                // printf("%d,%d,%d,%d\n", predSubblock.s0, predSubblock.s1, predSubblock.s2, predSubblock.s3);
                                // printf("%d,%d,%d,%d\n", predSubblock.s4, predSubblock.s5, predSubblock.s6, predSubblock.s7);
                                // printf("%d,%d,%d,%d\n", predSubblock.s8, predSubblock.s9, predSubblock.sa, predSubblock.sb);
                                // printf("%d,%d,%d,%d\n\n\n", predSubblock.sc, predSubblock.sd, predSubblock.se, predSubblock.sf);                    
                            }
                            //*/
                            
                            //*
                        }
                    }

                    barrier(CLK_LOCAL_MEM_FENCE); // Wait until all workitems finish their SATD computation

                    // PARALLEL REDUCTION
                    nPassesParallelSum = (int) log2( (float) min(16, cuWidth*cuHeight/16) ); // log2()
                    stride = itemsPerCuInSatd/2;
                    baseIdx = (lid/16)*16 + lid%16;
                    for(int pass=0; pass<nPassesParallelSum; pass++){
                        if(lid%16 < stride){
                            localSATD[baseIdx] = localSATD[baseIdx] + localSATD[baseIdx+stride];
                            stride = stride/2;
                        }    
                        barrier(CLK_LOCAL_MEM_FENCE);  
                    }
                }
            }


            //*/

            // Save SAD and SATD of current CU/mode in a __local buffer. We only access global memory when all SAD values are computed or all CUs
            if(lid==0){
                localSadEntireCtu[firstCu][mode] = localSAD[0];
                localSadEntireCtu[firstCu+1][mode] = localSAD[16];
                localSadEntireCtu[firstCu+2][mode] = localSAD[32];
                localSadEntireCtu[firstCu+3][mode] = localSAD[48];
                localSadEntireCtu[firstCu+4][mode] = localSAD[64];
                localSadEntireCtu[firstCu+5][mode] = localSAD[80];
                localSadEntireCtu[firstCu+6][mode] = localSAD[96];
                localSadEntireCtu[firstCu+7][mode] = localSAD[112];
                localSadEntireCtu[firstCu+8][mode] = localSAD[128];
                localSadEntireCtu[firstCu+9][mode] = localSAD[144];
                localSadEntireCtu[firstCu+10][mode] = localSAD[160];
                localSadEntireCtu[firstCu+11][mode] = localSAD[176];
                localSadEntireCtu[firstCu+12][mode] = localSAD[192];
                localSadEntireCtu[firstCu+13][mode] = localSAD[208];
                localSadEntireCtu[firstCu+14][mode] = localSAD[224];
                localSadEntireCtu[firstCu+15][mode] = localSAD[240];

                localSatdEntireCtu[firstCu][mode] = localSATD[0];
                localSatdEntireCtu[firstCu+1][mode] = localSATD[16];
                localSatdEntireCtu[firstCu+2][mode] = localSATD[32];
                localSatdEntireCtu[firstCu+3][mode] = localSATD[48];
                localSatdEntireCtu[firstCu+4][mode] = localSATD[64];
                localSatdEntireCtu[firstCu+5][mode] = localSATD[80];
                localSatdEntireCtu[firstCu+6][mode] = localSATD[96];
                localSatdEntireCtu[firstCu+7][mode] = localSATD[112];
                localSatdEntireCtu[firstCu+8][mode] = localSATD[128];
                localSatdEntireCtu[firstCu+9][mode] = localSATD[144];
                localSatdEntireCtu[firstCu+10][mode] = localSATD[160];
                localSatdEntireCtu[firstCu+11][mode] = localSATD[176];
                localSatdEntireCtu[firstCu+12][mode] = localSATD[192];
                localSatdEntireCtu[firstCu+13][mode] = localSATD[208];
                localSatdEntireCtu[firstCu+14][mode] = localSATD[224];
                localSatdEntireCtu[firstCu+15][mode] = localSATD[240];
            } 

        } // Finish current mode
    } // Finish current pair of CUs   
    
    // When all CUs are processed, we move the results into global buffer
    // if(lid < ALL_cusPerCtu[cuSizeIdx]*numPredictionModes*2){
       
    if(lid < 128*numPredictionModes*2){
        int nPassesMoveSadIntoGlobal = max(1, (int) ceil ((1.0*128*numPredictionModes*2)/wgSize));
        int idxInLocal, cu, mode, idxInGlobal;
        
        for(int pass=0; pass<nPassesMoveSadIntoGlobal; pass++){
            idxInLocal = pass*wgSize + lid;
            cu = idxInLocal / (numPredictionModes*2);
            mode = idxInLocal % (numPredictionModes*2);

            // Point to current CTU in global buffer
            idxInGlobal = ctuIdx*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES];
            // Point to current CU size in global buffer
            idxInGlobal    += ALL_stridedDistortionsPerCtu[cuSizeIdx] + cuIdxOffset*numPredictionModes*2;
            // Point to current CU and mode in global buffer
            idxInGlobal    += cu*numPredictionModes*2 + mode;

            if(cu < 128){
                SAD[idxInGlobal] = ( long ) localSadEntireCtu[cu][mode];
                SATD[idxInGlobal] = ( long ) localSatdEntireCtu[cu][mode];
                minSadHad[idxInGlobal] = (long) min(2*localSadEntireCtu[cu][mode], localSatdEntireCtu[cu][mode]);
            }
        }
    }
}