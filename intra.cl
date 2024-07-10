#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#pragma OPENCL EXTENSION cl_nv_compiler_options : enable

#endif

#include "mip_matrix.cl"
#include "kernel_aux_functions.cl"

#define BUFFER_SLOTS 2

// This kernel is used to fetch the reduced boundaries for all the blocks
// Each WG will process one CTU composed of a single CU size
// It works for all blocks with SizeId=2 and all alignments
__kernel void initBoundaries(__global short *referenceFrame, const int frameWidth, const int frameHeight, __global short *unified_redT, __global short *unified_redL, __global short *unified_refT, __global short *unified_refL, const int rep){
    // Each of these hold one row/columns of samples for the entire CTU
    __local short int refT[128], refL[128]; 
    // These buffers are used as temporary storage between computing reduced boundaries and moving them into global memory
    __local short int bufferGlobalRedT[2048], bufferGlobalRedL[2048]; // Maximum value. 1024 CUs 4x4 with reducedBoundary=2 each

    __local short int bufferGlobalRefT[MAX_CU_ROWS_PER_CTU][128], bufferGlobalRefL[MAX_CU_COLUMNS_PER_CTU][128];

//    for(int rep=0; rep<N_FRAMES; rep++){
        int gid = get_global_id(0);
        int wg = get_group_id(0);
        int lid = get_local_id(0);
        int wgSize = get_local_size(0);
        
        const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
        const short ctuRowsPerFrame = (short) ceil((float)frameHeight/128);
        const short nCTUs = ctuColumnsPerFrame*ctuRowsPerFrame;
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
            if((ctuY+cuY+cuHeight)>frameHeight){ // When sample lies outside the frame we do nothing
                // continue;
            }
            else if((ctuY+cuY)>0){ // Most general case, all references available
                refT[lid] = referenceFrame[(rep%BUFFER_SLOTS)*frameWidth*frameHeight + startCurrRow + cuX + lid%cuWidth]; // At this point, one row of samples is in the shared array. We must reduce it to obtain the redT for each CU
            }
            else if((ctuY+cuY)==0 && (ctuX+cuX)==0){ // CU is in the top-left corner of frame, no reference is available. Fill with predefined DC value
                refT[lid] = valueDC;
            }
            else if((ctuY+cuY)==0 && (ctuX+cuX)>0){ // CU is in the top edge of the frame, we use the left samples to pad the top boundaries
                refT[lid] = referenceFrame[(rep%BUFFER_SLOTS)*frameWidth*frameHeight + ctuX+cuX-1]; // Sample directly left of the first sample inside the CU is padded to top boundary
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
                
                int frameStride = (rep%BUFFER_SLOTS)*nCTUs*(ALL_TOTAL_CUS_SizeId12_PER_CTU*BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU*BOUNDARY_SIZE_Id0);

                unified_redT[frameStride + idx + pass*wgSize + lid] = bufferGlobalRedT[pass*wgSize + lid];
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
                
                int frameStride = (rep%BUFFER_SLOTS)*nCTUs*ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES];
                
                unified_refT[frameStride + ctuIdx*ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES] + ALL_stridedCompleteTopBoundaries[cuSizeIdx] + (currRow*cuColumnsPerCtu+cuInRow)*cuWidth + lid%cuWidth] = bufferGlobalRefT[currRow][cuInRow*cuWidth + lid%cuWidth];
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
            if((ctuY+cuY+cuHeight)>frameHeight){ // When sample lies outside the frame we do nothing
                //continue;
            }
            else if((ctuX+cuX)>0){ // Most general case, all neighboring samples are available
                refL[lid] = referenceFrame[(rep%BUFFER_SLOTS)*frameWidth*frameHeight + startCurrCol + (cuY+lid%cuHeight)*frameWidth]; // At this point, one row of samples is in the shared array. We must reduce it to obtain the redL for each CU
            }
            else if((ctuY+cuY)==0 && (ctuX+cuX)==0){ // CU is in the top-left corner of frame, no reference is available. Fill with predefined DC value
                refL[lid] = valueDC;
            }
            else if((ctuX+cuX)==0 && (ctuY+cuY)>0){ // CU is in the left edge of the frame, we use the top samples to pad the left boundaries
                refL[lid] = referenceFrame[(rep%BUFFER_SLOTS)*frameWidth*frameHeight + (ctuY+cuY-1)*frameWidth];  // Sample directly above of the first sample inside the CU is padded to left boundary
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
                
                int frameStride = (rep%BUFFER_SLOTS)*nCTUs*(ALL_TOTAL_CUS_SizeId12_PER_CTU*BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU*BOUNDARY_SIZE_Id0);

                unified_redL[frameStride + idx + pass*wgSize + lid] = bufferGlobalRedL[pass*wgSize + lid];
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
                    
                    int frameStride = (rep%BUFFER_SLOTS)*nCTUs*ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES];
                    
                    // Left boundaries are stored in global memory in CU-raster order. First the boundaries of CU_0, then CU_1, then CU_2, ... Since CUs are upsampled in raster order, this order improves the memory accesses
                    unified_refL[frameStride + ctuIdx*ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES] + ALL_stridedCompleteLeftBoundaries[cuSizeIdx] + currCuRow*cuColumnsPerCtu*cuHeight + currCuCol*cuHeight + currSample] = bufferGlobalRefL[currCuCol][currCuRow*cuHeight + currSample];
            }
        }
    //}
}

// This kernel is used to obtain the reduced prediction with all prediction modes
// The prediction of all prediction modes is stored in global memory and returned to the host
// Each WG will process one CTU composed of a single CU size
__kernel void MIP_ReducedPred(__global short *reducedPrediction, const int frameWidth, const int frameHeight, __global short* originalSamples, __global short *unified_redT, __global short *unified_redL, const int rep){
    
    // This buffer stores all predicted CUs inside the current CTU, with a single prediction mode
    // Each CU is processed by 64 workitems, where each workitem conducts the prediction of a single sample
    // When necessary, each workitem will process more than one CU
    // After all CUs are predicted with a single prediction mode, the buffer is moved into global memory and the next prediction mode is tested
    __local short reducedPredictedCtu[ 16384 ]; // When reducedPred=8x8 there are at most 256 CUs per CTU (256*8*8=16384). When reducedPred=4x4 there at exactly 1024 CUs (1024*4*4=16384)

    __local short upsampledPredictedCtu[128*128]; // used to store the entire CTU after upsampling, before computing distortion

    //for(int rep=0; rep<N_FRAMES; rep++){
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
        const short ctuRowsPerFrame = (short) ceil((float)frameHeight/128);
        const short nCTUs = ctuColumnsPerFrame*ctuRowsPerFrame;

        const short cuColumnsPerCtu = ALL_cuColumnsPerCtu[cuSizeIdx];
        const short cuRowsPerCtu = ALL_cuRowsPerCtu[cuSizeIdx];

        // CTU position inside the frame
        const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
        const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

        int boundaryStrideForCtu = ALL_TOTAL_CUS_SizeId12_PER_CTU*BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU*BOUNDARY_SIZE_Id0;
        int currCtuBoundariesIdx = ctuIdx * boundaryStrideForCtu;

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
        for(int m=0; m<totalPredictionModes; m++){
            
            short mode = m%(totalPredictionModes/2);
            short t = -(m/((totalPredictionModes/2))); // -1 because this value is used in select(), and select() only tests the MSB of the value
            short8 isTransp = (short8) (t,t,t,t,t,t,t,t);

            for(int pass=0; pass<nPasses; pass++){
                cuIdxInCtu = pass*cusPerPass + floor((float)lid/itemsPerCuInPrediction);

                // Point to current CU size. Even though CUs 4x4 have reducedBoundarySize=2, the ALL_stridedCusPerCtu points to the start of the current CU size and all previous sizes have reducedboundarySize=4
                
                int frameStride = (rep%BUFFER_SLOTS)*nCTUs*(ALL_TOTAL_CUS_SizeId12_PER_CTU*BOUNDARY_SIZE_Id12 + ALL_TOTAL_CUS_SizeId0_PER_CTU*BOUNDARY_SIZE_Id0);
                
                idx = frameStride + currCtuBoundariesIdx + ALL_stridedCusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY;

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

                // Fetch the coefficients from global
                uchar8 vectorizedCoeffs;
                if(reducedPredSize==8){ // SizeId==2
                    vectorizedCoeffs = vload8(0, &mipMatrix16x16[mode][sampleInCu][0]);
                    // Shift the coefficients to the right by 1 element, so that coeff 1 is in position [1]. Zero first coefficient beause it does not exist
                    uchar8 mask = (uchar8)(0,0,1,2,3,4,5,6); 
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

                    long int frameStride = (rep%BUFFER_SLOTS)*nCTUs*ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES];

                    reducedPrediction[frameStride + currCuPredictionIdx + sampleInCu] = reducedPredictedCtu[cuIdxInCtu*reducedPredSize*reducedPredSize + sampleInCu];
                }
                barrier(CLK_LOCAL_MEM_FENCE); // Wait until all predicted samples are moved. The next iteration overwrites the local prediction buffer
            }
        } // End of current mode
   // }
}

__kernel void upsampleDistortion(__global short *reducedPrediction, const int frameWidth, const int frameHeight,
#if ! MAX_PERFORMANCE_DIST
 __global long *SAD, __global long *SATD,
#endif
                                                                                                                     __global long *minSadHad, __global short* originalSamples, __global short *unified_refT, __global short *unified_refL, const int rep){

    // During upsampling, 128/32/16 workitems are assigned to conduct the processing of each CU (i.e., with wgSize=256 we process 2/8/16 CUs at once for SizeId=2/1/0)
    // We fetch the boundaries of these CUs depending on wgSize, upsample these CUs with all prediction modes to reuse the boundaries without extra memory access
    // Compute the distortion for these CUs with each prediction mode, then process the next CUs
    __local int localSAD[256], localSATD[256];

#if SIZEID==2
    __local short localReducedPrediction[2][REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2]; // At most, 2/8/16 CUs are processed simultaneously
    __local short localUpsampledPrediction[2][64*64]; // at most 2 CUs are predicted simultaneously, with a maximum dimension of 64x64
    __local short localOriginalSamples[2][64*64];   
    __local int localSadEntireCtu[ALL_MAX_CUS_PER_CTU_SizeId2][PREDICTION_MODES_ID2*2];
    __local int localSatdEntireCtu[ALL_MAX_CUS_PER_CTU_SizeId2][PREDICTION_MODES_ID2*2];
    __local int refT[2*64], refL[2*64]; // Complete boundaries of the CUs being processed
#elif SIZEID==1
    __local short localReducedPrediction[8][REDUCED_PRED_SIZE_Id1*REDUCED_PRED_SIZE_Id1]; // At most, 2/8/16 CUs are processed simultaneously
    __local short localUpsampledPrediction[8][32*4]; // at most 8 CUs are predicted simultaneously, with a maximum dimension of 32x4 or 4x32
    __local short localOriginalSamples[8][32*4];   
    __local int localSadEntireCtu[ALL_MAX_CUS_PER_CTU_SizeId1][PREDICTION_MODES_ID1*2];
    __local int localSatdEntireCtu[ALL_MAX_CUS_PER_CTU_SizeId1][PREDICTION_MODES_ID1*2];
    __local int refT[8*32], refL[8*32]; // Complete boundaries of the CUs being processed
#else // SIZEID==0
    __local short localReducedPrediction[16][REDUCED_PRED_SIZE_Id0*REDUCED_PRED_SIZE_Id0]; // At most, 2/8/16 CUs are processed simultaneously
    __local short localUpsampledPrediction[16][4*4]; // at most 16 CUs are predicted simultaneously, with a maximum dimension of 4x4
    __local short localOriginalSamples[16][4*4];   
    __local int localSadEntireCtu[128][PREDICTION_MODES_ID0*2];
    __local int localSatdEntireCtu[128][PREDICTION_MODES_ID0*2];
    __local int refT[2], refL[2]; // These are not  relevant for SizeId=0. Declared here to avoid errors
#endif

//for(int rep=0; rep<N_FRAMES; rep++){


    #if SIZEID==2
        // printf("SizeId=%d\n", SIZEID);
        const int sizeId = 2;
        const int NUM_CU_SIZES_CURR_ID = NUM_CU_SIZES_SizeId2;
        const int firstCuSizeForId = 0;
        const int numPredictionModes = PREDICTION_MODES_ID2;
        const char reducedBoundarySize = BOUNDARY_SIZE_Id2;
        const char reducedPredSize = REDUCED_PRED_SIZE_Id2;
        const int samplesInSmallerCu = 128; // This is based on the smaller CU, 8*16=128
        const int log2samplesInSmallerCu = 7; // used for parallel reduction
        const int cusInParallel = 2; // We have at most 256 workitems per WG, and will use 128 workitems per CU. Therefore, 2 CUs simultaneously
        const int largestPossibleSide = 64; // This depends on the largest possible side for CUs in this SizeId
    #elif SIZEID==1
        // printf("SizeId=%d\n", SIZEID);
        const int sizeId = 1;
        const int NUM_CU_SIZES_CURR_ID = NUM_CU_SIZES_SizeId1;
        const int firstCuSizeForId = FIRST_SizeId1;
        const int numPredictionModes = PREDICTION_MODES_ID1;
        const char reducedBoundarySize = BOUNDARY_SIZE_Id1;
        const char reducedPredSize = REDUCED_PRED_SIZE_Id1;
        const int samplesInSmallerCu = 32; // This is based on the smaller CU, 4*8=32
        const int log2samplesInSmallerCu = 5; // used for parallel reduction
        const int cusInParallel = 8; // We have at most 256 workitems per WG, and will use 32 workitems per CU. Therefore, 8 CUs simultaneously
        const int largestPossibleSide = 32; // This depends on the largest possible side for CUs in this SizeId
    #else // SIZEID == 0
        // printf("SizeId=%d\n", 0);
        const int sizeId = 0;
        const int NUM_CU_SIZES_CURR_ID = NUM_CU_SIZES_SizeId0*8; // For SizeId=0, we use 8 WGs to process each CU. It is easier to consider 8 CU sizes for the math
        const int firstCuSizeForId = FIRST_SizeId0;
        const int numPredictionModes = PREDICTION_MODES_ID0;
        const char reducedBoundarySize = BOUNDARY_SIZE_Id0;
        const char reducedPredSize = REDUCED_PRED_SIZE_Id0;
        const int samplesInSmallerCu = 16; // This is based on the smaller CU, 4*4=16
        const int log2samplesInSmallerCu = 4; // used for parallel reduction
        const int cusInParallel = 16; // We have at most 256 workitems per WG, and will use 16 workitems per CU. Therefore, 16 CUs simultaneously
        const int largestPossibleSide = 4; // This depends on the largest possible side for CUs in this SizeId
    #endif

        int gid = get_global_id(0);
        int wg = get_group_id(0);
        int lid = get_local_id(0);
        int wgSize = get_local_size(0);

        const short ctuIdx = wg/NUM_CU_SIZES_CURR_ID;
        const short cuSizeIdx = select(wg%NUM_CU_SIZES_CURR_ID + firstCuSizeForId, ALL_AL_4x4, sizeId==0);
        const short cuWidth = ALL_widths[cuSizeIdx];
        const short cuHeight = ALL_heights[cuSizeIdx];
        const short nCusInCtu = select((int) ALL_cusPerCtu[cuSizeIdx], 128, sizeId==0); 

        const short ctuColumnsPerFrame = (short) ceil((float)frameWidth/128);
        const short ctuRowsPerFrame = (short) ceil((float)frameHeight/128);
        const short nCTUs = ctuColumnsPerFrame*ctuRowsPerFrame;
        
        const short cuColumnsPerCtu = ALL_cuColumnsPerCtu[cuSizeIdx];
        const short cuRowsPerCtu = ALL_cuRowsPerCtu[cuSizeIdx];

        // CTU position inside the frame
        const short ctuX = 128 * (ctuIdx%ctuColumnsPerFrame);  
        const short ctuY = 128 * (ctuIdx/ctuColumnsPerFrame);

        int boundaryStrideForCtu = ALL_cusPerCtu[cuSizeIdx]*LARGEST_RED_BOUNDARY; // Each CU occupy LARGEST_RED_BOUNDARY (=4) positions in the reduced boundaries buffers
        int currCtuBoundariesIdx = ctuIdx * boundaryStrideForCtu;

        const int upsamplingHorizontal = cuWidth / reducedPredSize;
        const int upsamplingVertical = cuHeight / reducedPredSize;

        const int log2UpsamplingHorizontal = (int) log2((float) upsamplingHorizontal);
        const int roundingOffsetHorizontal = 1 << (log2UpsamplingHorizontal - 1);

        const int log2UpsamplingVertical = (int) log2((float) upsamplingVertical);
        const int roundingOffsetVertical = 1 << (log2UpsamplingVertical - 1);
        
        // ######################################################################
        //      Variables shared for horizontal and vertical interpolation
        int xPosInCu, yPosInCu, xPosInCtu, yPosInCtu, xPosInFrame, yPosInFrame, idx;
        int valueBefore, valueAfter, beforeIdx, afterIdx;
        int isMiddle;
        int offsetInStride;
        int itemsPerCuInUpsampling;
        int itemsPerCuInFetchOriginal;
        int itemsPerCuInSatd;
        
        // During upsampling, 128/32/16 workitems are assigned to conduct the processing of each CU (i.e., with wgSize=256 we process 2/8/16 CUs at once for SizeId=2/1/0)
        // We fetch the boundaries of these CUs depending on wgSize, upsample these CUs with all prediction modes to reuse the boundaries without extra memory access
        // Compute the distortion for these CUs with each prediction mode, then process the next CUs
    #if SIZEID==2
        // Specifically for CUs 4x4 with SizeId=0. For other sizes it is zero
        const int cuIdxOffset = 0;
        const int offsetInCtuY = 0;
    #elif SIZEID==1
        // Specifically for CUs 4x4 with SizeId=0. For other sizes it is zero
        const int cuIdxOffset = 0;
        const int offsetInCtuY = 0;
    #else // SIZEID==0
        // Specifically for CUs 4x4 with SizeId=0. For other sizes it is zero
        const int cuIdxOffset = 128 * (wg%8); // Each WG processes 12.5% of the CTU, therefore we must correct the first sample where each WG start
        const int offsetInCtuY = 4*(cuIdxOffset/cuColumnsPerCtu);
    #endif

        

        // Each CU will be upsampled using 128/32/16 workitems, irrespective of the CU size
        // We will process 2/8/16 CUs simultaneously when wgSize=256
        // CUs with more samples will require multiple passes (i.e., CUs larger than 8x16 and 16x8 for SizeId=2)
        itemsPerCuInUpsampling = samplesInSmallerCu;
        itemsPerCuInFetchOriginal = samplesInSmallerCu;

        for(int firstCu = 0; firstCu < nCusInCtu; firstCu += wgSize/itemsPerCuInUpsampling){
            int cuIdxInIteration = lid/itemsPerCuInUpsampling; // This represents if the current CU equals firstCU, firstCU+1, firstCU+2, ...
            // The Offset is different from zero only when SIzeId=0
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
    #if SIZEID>0
                xPosInCtu = ALL_X_POS[cuSizeIdx][currCu] + xPosInCu;
                yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu] + yPosInCu;
    #else // SIZEID==0
                xPosInCtu = 4*(currCu%cuColumnsPerCtu) + xPosInCu;
                yPosInCtu = 4*(currCu/cuColumnsPerCtu) + yPosInCu;
    #endif
                xPosInFrame = ctuX + xPosInCtu;
                yPosInFrame = ctuY + yPosInCtu;
                
                int frameStride = (rep%BUFFER_SLOTS)*frameWidth*frameHeight;

                if(yPosInFrame<frameHeight) // When sample lies outside the frame we do nothing
                    localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] = originalSamples[frameStride + yPosInFrame*frameWidth + xPosInFrame];
            }
            

            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      FETCH THE BOUNDARIES REQUIRED FOR UPSAMPLING

    // SIZEID=0 Does not require upsampling nor the boundaries
    #if SIZEID>0        
            int topBoundariesIdx, leftBoundariesIdx;

            // Points to the current CTU boundaries
            topBoundariesIdx = ctuIdx * ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES];
            leftBoundariesIdx = ctuIdx * ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES];

            // Points to the current CU boundaries
            topBoundariesIdx  += ALL_stridedCompleteTopBoundaries[cuSizeIdx] + currCu*cuWidth;
            leftBoundariesIdx += ALL_stridedCompleteLeftBoundaries[cuSizeIdx] + currCu*cuHeight;
            
            int frameStrideTop = (rep%BUFFER_SLOTS)*nCTUs*ALL_stridedCompleteTopBoundaries[ALL_NUM_CU_SIZES];
            int frameStrideLeft = (rep%BUFFER_SLOTS)*nCTUs*ALL_stridedCompleteLeftBoundaries[ALL_NUM_CU_SIZES];

            // Fetch TOP boundaries
            if(lid%itemsPerCuInUpsampling < cuWidth){
                refT[cuIdxInIteration*largestPossibleSide + lid%itemsPerCuInUpsampling] =  unified_refT[frameStrideTop + topBoundariesIdx + lid%itemsPerCuInUpsampling];
            }

            // Fetch LEFT boundaries
            // TODO: We only need the reduced left boundaries. It reduces the number of global memory reads at this point. For the top boundaries we will need the full references
            if(lid%itemsPerCuInUpsampling < cuHeight){
                refL[cuIdxInIteration*largestPossibleSide + lid%itemsPerCuInUpsampling] =  unified_refL[frameStrideLeft + leftBoundariesIdx + lid%itemsPerCuInUpsampling];
            }
            
            /*  TRACE THE BOUNDARIES FOR THE CURRENT CU
            //if(0 && ctuIdx==0 && cuSizeIdx==_16x16 && currCu==45 && lid==128){
            //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && lid%itemsPerCuInUpsampling==0){
            if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && lid%itemsPerCuInUpsampling==0){
                //printf("\n\n\n\n\n OI %d %d\n\n\n\n\n\n", refT[0], refL[0]);
                printf("TOP BOUNDARIES,CTU=%d,WH=%dx%d,CU=%d\n", ctuIdx, cuWidth, cuHeight, currCu);
                for(int i=0; i<cuWidth; i++){
                    printf("%d,", refT[(currCu%cusInParallel)*largestPossibleSide + i]);
                }
                printf("\n");
                printf("LEFT BOUNDARIES,CTU=%d,WH=%dx%d,CU=%d\n", ctuIdx, cuWidth, cuHeight, currCu);
                for(int i=0; i<cuHeight; i++){
                    printf("%d,", refL[(currCu%cusInParallel)*largestPossibleSide + i]);
                }
                printf("\n");
            }
            //*/
    #endif

            int idxCurrCuAndMode = 0;
            // Now we do the upsampling for all prediction modes of the current 2/8/16 CUs
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

                long int frameStride = (rep%BUFFER_SLOTS)*nCTUs*ALL_stridedPredictionsPerCtu[ALL_NUM_CU_SIZES];

                if(lid%itemsPerCuInUpsampling < (reducedPredSize*reducedPredSize)){
                    localReducedPrediction[cuIdxInIteration][lid%itemsPerCuInUpsampling] = reducedPrediction[frameStride + idxCurrCuAndMode + lid%itemsPerCuInUpsampling];
                }

                barrier(CLK_LOCAL_MEM_FENCE); // Wait until the whole buffer for reduced prediction is filled


                /* Trace reduced prediction fetched from global memory
                if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_8x16_G4 && currCu==11 && mode==0 && lid%itemsPerCuInUpsampling==0){
                //if(1 && ctuIdx==16 && cuSizeIdx==ALL_NA_16x16_G3 && currCu==14 && mode==0 && lid%itemsPerCuInUpsampling==0){
                    printf("REDUCED PREDICTION,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
                    for(int i=0; i<reducedPredSize; i++){
                        for(int j=0; j<reducedPredSize; j++){
                            printf("%d,", localReducedPrediction[cuIdxInIteration][i*reducedPredSize+j]);
                        }
                        printf("\n");
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                //*/


                // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                //
                //      START WITH HORIZONTAL UPSAMPLING...
    // SIZEID=0 Does not require upsampling
    #if SIZEID>0
                int nPassesHorizontalUpsampling = max(1, (cuWidth*reducedPredSize)/itemsPerCuInUpsampling);

                for(int pass=0; pass<nPassesHorizontalUpsampling; pass++){
                    int idx = pass*itemsPerCuInUpsampling + lid%itemsPerCuInUpsampling;
                    xPosInCu = idx%cuWidth;
                    yPosInCu = (idx/cuWidth)*upsamplingVertical + upsamplingVertical-1;
                    xPosInCtu = ALL_X_POS[cuSizeIdx][currCu]; // (currCu%cuColumnsPerCtu)*cuWidth + xPosInCu;
                    yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu]; // (currCu/cuColumnsPerCtu)*cuHeight + yPosInCu;

                    isMiddle = xPosInCu>=upsamplingHorizontal; // In this case, the left boundary is not used
                    offsetInStride = xPosInCu%upsamplingHorizontal+1; // Position inside one window where samples are being interpolated. BeforeReference has stride=0, first interpolated sample has stride=1            }

                    if(lid%itemsPerCuInUpsampling < cuWidth*reducedPredSize){ // Sometimes the reduced rpediction has the same dimension as the final CU in one direction, therefore we only copy
                        // For the first couple of sample columns, the "before" reference is the refL buffer
                        if(isMiddle == 0){
                            // Predicted value that is before the current sample
                            valueBefore = refL[cuIdxInIteration*largestPossibleSide + yPosInCu];
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
                        valueBefore = refT[cuIdxInIteration*largestPossibleSide + xPosInCu];
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

    #endif
                
                // At this point the upsampling for the current mode is complete. We can compute the distortion...
                
                // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                //
                //      COMPUTE SAD FOR THE CURRENT CU

                // Here, each workitem will compute the SAD at one or more position, and accumulate the result in localSAD[lid]
                // At the end we must reduce the valeus between 0-127 and 128-255 to obtain the final SAD of each CU
                localSAD[lid] = 0;
                localSATD[lid] = 0;

                int nPassesForSad = nPassesOriginalFetch;

                for(int pass=0; pass<nPassesForSad; pass++){
                    idx = pass*itemsPerCuInFetchOriginal + lid%itemsPerCuInFetchOriginal;
                    xPosInCu = idx%cuWidth;
                    yPosInCu = idx/cuWidth;
    // SIZEID=0 Does not employ upsampling and the distortion is computed from the reduced prediction
    #if SIZEID>0    
                    localSAD[lid] += (int) abs(localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] - localUpsampledPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu]);
    #else // SIZEID==0
                    localSAD[lid] += (int) abs(localOriginalSamples[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu] - localReducedPrediction[cuIdxInIteration][yPosInCu*cuWidth + xPosInCu]);
    #endif
                }            

                // if(1 && ctuIdx==16 && cuSizeIdx==_64x64 && currCu==3 && lid%128==0 && mode==0){
                //     printf("SERIAL SAD FOR,CTU=%d,WH=%dx%d,CU=%d,MODE=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode);
                //     for(int i=0; i<128; i++){
                //         printf("%d,", localSAD[lid+i]);
                //     }
                //     printf("\n");
                // }

                // PARALLEL REDUCTION
                int nPassesParallelSum = log2samplesInSmallerCu;
                int stride = samplesInSmallerCu/2;
                int baseIdx = (lid/samplesInSmallerCu)*samplesInSmallerCu + lid%samplesInSmallerCu;
                for(int pass=0; pass<nPassesParallelSum; pass++){
                    barrier(CLK_LOCAL_MEM_FENCE);  
                    if(lid%samplesInSmallerCu < stride){
                        localSAD[baseIdx] = localSAD[baseIdx] + localSAD[baseIdx+stride];
                        stride = stride/2;
                    }    
                }


                /* TRACE SAD
                if(1 && ctuIdx==0 && cuSizeIdx==_64x64 && currCu==0 && lid%samplesInSmallerCu==0){
                    printf("SAD,CTU=%d,WH=%dx%d,CU=%d,MODE=%d,SAD=%d\n", ctuIdx, cuWidth, cuHeight, currCu, mode, localSAD[(cuIdxInIteration%cusInParallel)*samplesInSmallerCu]);
                    // printf("OI,localSAD[lid]=%d\n", localSAD[lid]);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                //*/      
                
                localSATD[lid] = 0;

                // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
                //
                //      COMPUTE SATD FOR THE CURRENT CU
                int idxSubblock;
                int nPassesForSatd = max(1,nPassesOriginalFetch/16);
                itemsPerCuInSatd = min(samplesInSmallerCu, (cuWidth*cuHeight)/16);
                int subblockX, subblockY;
                short16 origSubblock, predSubblock;

                // TODO: Here we can improve performance by focusing the work on fewer warps/wavefronts
                // Here, each workitem will compute the SATD for one or more subblocks 4x4, and accumulate the results in __local localSATD
                if((lid%samplesInSmallerCu) < itemsPerCuInSatd){
                    for(int pass=0; pass<nPassesForSatd; pass++){
                        idxSubblock = pass*itemsPerCuInSatd + lid%samplesInSmallerCu;
                        subblockX = (idxSubblock%(cuWidth/4))<<2;
                        subblockY = (idxSubblock/(cuWidth/4))<<2;
                        idx = subblockY*(cuWidth/4) + subblockX/4;


                        // 1st row                    
                        origSubblock.lo.lo = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                        // SIZEID=0 Does not employ upsampling and the distortion is computed from the reduced prediction
                        #if SIZEID>0
                            predSubblock.lo.lo = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                        #else // SIZEID==0
                            predSubblock.lo.lo = vload4(idx, localReducedPrediction[cuIdxInIteration]);
                        #endif
                        // 2nd row
                        idx += cuWidth/4;
                        origSubblock.lo.hi = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                        #if SIZEID>0
                            predSubblock.lo.hi = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                        #else // SIZEID==0
                            predSubblock.lo.hi = vload4(idx, localReducedPrediction[cuIdxInIteration]);
                        #endif
                        // 3rd row
                        idx += cuWidth/4;
                        origSubblock.hi.lo = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                        #if SIZEID>0
                            predSubblock.hi.lo = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                        #else // SIZEID==0
                            predSubblock.hi.lo = vload4(idx, localReducedPrediction[cuIdxInIteration]);
                        #endif
                        // 4th row
                        idx += cuWidth/4;
                        origSubblock.hi.hi = vload4(idx, localOriginalSamples[cuIdxInIteration]);
                        #if SIZEID>0
                            predSubblock.hi.hi = vload4(idx, localUpsampledPrediction[cuIdxInIteration]);
                        #else // SIZEID==0
                            predSubblock.hi.hi = vload4(idx, localReducedPrediction[cuIdxInIteration]);
                        #endif

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
                    }
                }

                // PARALLEL REDUCTION
                nPassesParallelSum = (int) log2( (float) min(samplesInSmallerCu, cuWidth*cuHeight/16) ); // log2()
                stride = itemsPerCuInSatd/2;
                baseIdx = (lid/samplesInSmallerCu)*samplesInSmallerCu + lid%samplesInSmallerCu;
                for(int pass=0; pass<nPassesParallelSum; pass++){
                    barrier(CLK_LOCAL_MEM_FENCE);  
                    if(lid%samplesInSmallerCu < stride){
                        localSATD[baseIdx] = localSATD[baseIdx] + localSATD[baseIdx+stride];
                        stride = stride/2;
                    }    
                }

                // Wait until all positions of localSATD are updated before offloading to global
                barrier(CLK_LOCAL_MEM_FENCE);

                // Save SAD and SATD of current CU/mode in a __local buffer. We only access global memory when all SAD values are computed or all CUs
                if(lid==0){
    #if SIZEID==2
                    localSadEntireCtu[firstCu][mode] = localSAD[0];
                    localSadEntireCtu[firstCu+1][mode] = localSAD[128];

                    localSatdEntireCtu[firstCu][mode] = localSATD[0];
                    localSatdEntireCtu[firstCu+1][mode] = localSATD[128];
    #elif SIZEID==1
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
    #else // SIZEID==0
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
    #endif
                } 
            } // Finish current mode
        } // Finish current pair of CUs   
        
        barrier(CLK_LOCAL_MEM_FENCE); // Wait untill all distortion values are copied to local CTU buffer before offloading to global
        // When all CUs are processed, we move the results into global buffer
        // SIZEID=0 Uses has a trick to use 8 WGs per CTU, using processing only 1/8 of the CUs each
    #if SIZEID>0    
        if(lid < ALL_cusPerCtu[cuSizeIdx]*numPredictionModes*2){
    #else
        if(lid < 128*numPredictionModes*2){
    #endif
            int nPassesMoveSadIntoGlobal;
            #if SIZEID>0
                nPassesMoveSadIntoGlobal = max(1, (int)ceil((1.0*ALL_cusPerCtu[cuSizeIdx]*numPredictionModes*2)/wgSize));
            #else // SIZEID==0
                nPassesMoveSadIntoGlobal = max(1, (int)ceil((1.0*128*numPredictionModes*2)/wgSize));
            #endif
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

                int frameStride = (rep%BUFFER_SLOTS)*nCTUs*ALL_stridedDistortionsPerCtu[ALL_NUM_CU_SIZES];

                idxInGlobal += frameStride;

    // SIZEID=0 Uses has a trick to use 8 WGs per CTU, using processing only 1/8 of the CUs each
    #if SIZEID>0
                if(cu < ALL_cusPerCtu[cuSizeIdx]){
    #else            
                if(cu < 128){
    #endif

    #if ! MAX_PERFORMANCE_DIST // When MAX PERFORMANCE is enabled, the individual SAD and SATD values are not returned to host
                    SAD[idxInGlobal] = ( long ) localSadEntireCtu[cu][mode];
                    SATD[idxInGlobal] = ( long ) localSatdEntireCtu[cu][mode];
    #endif
                    // minSadHad is always returned
                    minSadHad[idxInGlobal] = (long) min(2*localSadEntireCtu[cu][mode], localSatdEntireCtu[cu][mode]);
                }
            } // Close for nPasses
        } // Close if lid < const
   // } // Close for N_FRAMES
} // Close kernel

// This kernel is used to fetch the original samples and apply a low-pass filter
// The filtered samples are used as references during prediction
__kernel void filterFrame_2d(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    unsigned int convKernel[3][3];
    
    convKernel[0][0] = convKernelLib[kernelIdx][0][0];
    convKernel[0][1] = convKernelLib[kernelIdx][0][1];
    convKernel[0][2] = convKernelLib[kernelIdx][0][2];
    convKernel[1][0] = convKernelLib[kernelIdx][1][0];
    convKernel[1][1] = convKernelLib[kernelIdx][1][1];
    convKernel[1][2] = convKernelLib[kernelIdx][1][2];
    convKernel[2][0] = convKernelLib[kernelIdx][2][0];
    convKernel[2][1] = convKernelLib[kernelIdx][2][1];
    convKernel[2][2] = convKernelLib[kernelIdx][2][2];

    unsigned int scale = convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];
    unsigned int topScale = convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];
    unsigned int bottomScale = convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2];
    unsigned int leftScale = convKernel[0][1]+convKernel[0][2]+convKernel[1][1]+convKernel[1][2]+convKernel[2][1]+convKernel[2][2];
    unsigned int rightScale = convKernel[0][0]+convKernel[0][1]+convKernel[1][0]+convKernel[1][1]+convKernel[2][0]+convKernel[2][1];
    unsigned int topLeftScale = convKernel[1][1]+convKernel[1][2]+convKernel[2][1]+convKernel[2][2];
    unsigned int topRightScale = convKernel[1][0]+convKernel[1][1]+convKernel[2][0]+convKernel[2][1];
    unsigned int bottomLeftScale = convKernel[0][1]+convKernel[0][2]+convKernel[1][1]+convKernel[1][2];
    unsigned int bottomRightScale = convKernel[0][0]+convKernel[0][1]+convKernel[1][0]+convKernel[1][1];
    unsigned int isTop=0, isBottom=0, isLeft=0, isRight=0;

    __local unsigned short filteredCTU[128*128];

    int halfCtuColumns = ceil(frameWidth/128.0);
    int halfCtuRows = ceil(frameHeight/64.0);

    
    // int ctuIdx = wg; //wg;
    int halfCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int halfCtuX = (halfCtuIdx % halfCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int halfCtuY = (halfCtuIdx / halfCtuColumns)*64;


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 sample around the input
    __local short origHalfCTU[130*66];
    __local short filteredHalfCTU[128*64];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*64/wgSize;
    int rowsPerPass = wgSize/128;
    int g_halfCtuBaseIdx = halfCtuY*frameWidth + halfCtuX;
    int haloOffset = 130+1;
    int l_ctuStride = 130;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
    }

    // Fetch the halo
   
    // Upper and lower edges: WIs in 0:127 fetch the top, WIs 128:255 fetch the bottom
    int currRow = (lid/128)*(65); // Either first or last row of halo
   
    if(((g_halfCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)>0) && ((g_halfCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origHalfCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128];
    }
        

    // Left and right edges: WIs 0 and 1 fetch the first row of both columns, WIs 2 and 3 fetch the second row, and so on...
    currRow = lid/2; // 2 WIs per row
    int currCol = lid%2; // Either first or last column
    if((lid<(2*64)) && ((g_halfCtuBaseIdx-1 + currRow*frameWidth + currCol*(l_ctuStride-1))>0) && ((g_halfCtuBaseIdx-1 + currRow*frameWidth + currCol*(l_ctuStride-1))<(frameWidth*frameHeight))){
        // skip TL corner                                                              left neighbor col |
        origHalfCTU[l_ctuStride + currRow*l_ctuStride + currCol*(l_ctuStride-1)] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx-1 + currRow*frameWidth + currCol*(l_ctuStride-1)];
    }
        

    // Top-left, top-right, bottom-left, bottom-right
    if(lid==0){
        if((g_halfCtuBaseIdx - frameWidth - 1)>0)
            origHalfCTU[0]          = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - frameWidth - 1]; // TL
        if((g_halfCtuBaseIdx - frameWidth + 128)>0)
            origHalfCTU[129]        = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - frameWidth + 128]; // TR
        if((g_halfCtuBaseIdx + 64*frameWidth -1)<(frameWidth*frameHeight))
            origHalfCTU[65*130]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 64*frameWidth -1]; // BL
        if((g_halfCtuBaseIdx + 64*frameWidth + 128)<(frameWidth*frameHeight))
            origHalfCTU[65*130+129] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 64*frameWidth + 128]; // BR
    }
    

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);


    int nPassesFilter = (128*64)/wgSize;

    unsigned int result;
    int mask[3][3];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    unsigned int currScale = scale;


    haloOffset = 130+1;
    l_ctuStride = 130;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){
        mask[0][0] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - l_ctuStride - 1];
        mask[0][1] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - l_ctuStride - 0];
        mask[0][2] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - l_ctuStride + 1];

        mask[1][0] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - 1];
        mask[1][1] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - 0];
        mask[1][2] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset + 1];

        mask[2][0] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset + l_ctuStride - 1];
        mask[2][1] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset + l_ctuStride - 0];
        mask[2][2] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset + l_ctuStride + 1];

        // We only have to deal with the left and top edges. The bottom and right edges are never used as references
        // Put a zero on the samples of the mask that lie outside the frame
        if(halfCtuY+currRow==0){ // Top of frame
            mask[0][0] = 0;
            mask[0][1] = 0;
            mask[0][2] = 0;
            currScale = topScale;
            isTop = 1;
        }
        if(halfCtuY+currRow==frameHeight-1){ // Bottom of frame
            mask[2][0] = 0;
            mask[2][1] = 0;
            mask[2][2] = 0;
            currScale = bottomScale;
            isBottom = 1;
        }
        if(halfCtuX+currCol==0){ // Left of frame
            mask[0][0] = 0;
            mask[1][0] = 0;
            mask[2][0] = 0;
            currScale = leftScale;
            isLeft = 1;
        }
        if(halfCtuX+currCol==frameWidth-1){ // Right of frame
            mask[0][2] = 0;
            mask[1][2] = 0;
            mask[2][2] = 0;
            currScale = rightScale;
            isRight = 1;
        }

        // Check explicitly if the block is on the top-left or top-right corners to adjust the scale. The coefficients are already adjusted by entering the 2 conditionals
        unsigned int isTopLeft = (isTop && isLeft);
        unsigned int isTopRight = (isTop && isRight);
        unsigned int isBottomLeft = (isBottom && isLeft);
        unsigned int isBottomRight = (isBottom && isRight);
        currScale = select(currScale, topLeftScale, isTopLeft);
        currScale = select(currScale, topRightScale, isTopRight);
        currScale = select(currScale, bottomLeftScale, isBottomLeft);
        currScale = select(currScale, bottomRightScale, isBottomRight);

        result = 0;

        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                result += mask[i][j]*convKernel[i][j];
            }
        }

        result = (result + currScale/2)/currScale;

        filteredHalfCTU[currRow*128 + currCol] = result;

        currRow += rowsPerPass;
        currScale = scale;
        isTop = 0;
        isBottom = 0;
        isLeft = 0;
        isRight = 0;

    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      OFFLOAD FILTERED SAMPELS INTO GLOBAL MEMORY AGAIN
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    barrier(CLK_LOCAL_MEM_FENCE);


    // if(lid==0 && halfCtuIdx==31){
    //     for(int h=0; h<30; h++){
    //         for(int w=0; w<128; w++){
    //             printf("%d,", filteredHalfCTU[h*280+w]);
    //         }
    //         printf("\n");
    //     }
    // }

    
    int rowsRemaininig = min(64, frameHeight - halfCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    rowsPerPass = wgSize/128;
    haloOffset = 130+1;
    l_ctuStride = 130;


    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        // filteredFrame[g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128];
        filteredFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = filteredHalfCTU[pass*rowsPerPass*128 + (lid/128)*128 + lid%128];
    }


}

__kernel void filterFrame_2d_float(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    float convKernel[3][3];
    
    convKernel[0][0] = (float) convKernelLib[kernelIdx][0][0];
    convKernel[0][1] = (float) convKernelLib[kernelIdx][0][1];
    convKernel[0][2] = (float) convKernelLib[kernelIdx][0][2];
    convKernel[1][0] = (float) convKernelLib[kernelIdx][1][0];
    convKernel[1][1] = (float) convKernelLib[kernelIdx][1][1];
    convKernel[1][2] = (float) convKernelLib[kernelIdx][1][2];
    convKernel[2][0] = (float) convKernelLib[kernelIdx][2][0];
    convKernel[2][1] = (float) convKernelLib[kernelIdx][2][1];
    convKernel[2][2] = (float) convKernelLib[kernelIdx][2][2];

    float scale = convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];
    float topScale = convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];
    float bottomScale = convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2];
    float leftScale = convKernel[0][1]+convKernel[0][2]+convKernel[1][1]+convKernel[1][2]+convKernel[2][1]+convKernel[2][2];
    float rightScale = convKernel[0][0]+convKernel[0][1]+convKernel[1][0]+convKernel[1][1]+convKernel[2][0]+convKernel[2][1];
    float topLeftScale = convKernel[1][1]+convKernel[1][2]+convKernel[2][1]+convKernel[2][2];
    float topRightScale = convKernel[1][0]+convKernel[1][1]+convKernel[2][0]+convKernel[2][1];
    float bottomLeftScale = convKernel[0][1]+convKernel[0][2]+convKernel[1][1]+convKernel[1][2];
    float bottomRightScale = convKernel[0][0]+convKernel[0][1]+convKernel[1][0]+convKernel[1][1];
    unsigned int isTop=0, isBottom=0, isLeft=0, isRight=0;

    int halfCtuColumns = ceil(frameWidth/128.0);
    int halfCtuRows = ceil(frameHeight/64.0);

    // int ctuIdx = wg; //wg;
    int halfCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int halfCtuX = (halfCtuIdx % halfCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int halfCtuY = (halfCtuIdx / halfCtuColumns)*64;


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 sample around the input
    __local short origHalfCTU[130*66];
    __local short filteredHalfCTU[128*64];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*64/wgSize;
    int rowsPerPass = wgSize/128;
    int g_halfCtuBaseIdx = halfCtuY*frameWidth + halfCtuX;
    int haloOffset = 130+1;
    int l_ctuStride = 130;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
    }


    // Fetch the halo
   
    // Upper and lower edges: WIs in 0:127 fetch the top, WIs 128:255 fetch the bottom
    int currRow = (lid/128)*(65); // Either first or last row of halo
   
    if(((g_halfCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)>0) && ((g_halfCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origHalfCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128];
    }
        

    // Left and right edges: WIs 0 and 1 fetch the first row of both columns, WIs 2 and 3 fetch the second row, and so on...
    currRow = lid/2; // 2 WIs per row
    int currCol = lid%2; // Either first or last column
    if((lid<(2*64)) && ((g_halfCtuBaseIdx-1 + currRow*frameWidth + currCol*(l_ctuStride-1))>0) && ((g_halfCtuBaseIdx-1 + currRow*frameWidth + currCol*(l_ctuStride-1))<(frameWidth*frameHeight))){
        // skip TL corner                                                              left neighbor col |
        origHalfCTU[l_ctuStride + currRow*l_ctuStride + currCol*(l_ctuStride-1)] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx-1 + currRow*frameWidth + currCol*(l_ctuStride-1)];
    }
        

    // Top-left, top-right, bottom-left, bottom-right
    if(lid==0){
        if((g_halfCtuBaseIdx - frameWidth - 1)>0)
            origHalfCTU[0]          = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - frameWidth - 1]; // TL
        if((g_halfCtuBaseIdx - frameWidth + 128)>0)
            origHalfCTU[129]        = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - frameWidth + 128]; // TR
        if((g_halfCtuBaseIdx + 64*frameWidth -1)<(frameWidth*frameHeight))
            origHalfCTU[65*130]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 64*frameWidth -1]; // BL
        if((g_halfCtuBaseIdx + 64*frameWidth + 128)<(frameWidth*frameHeight))
            origHalfCTU[65*130+129] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 64*frameWidth + 128]; // BR
    }
    

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);


    int nPassesFilter = (128*64)/wgSize;

    float result;
    int mask[3][3];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    float currScale = scale;


    haloOffset = 130+1;
    l_ctuStride = 130;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){
        mask[0][0] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - l_ctuStride - 1];
        mask[0][1] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - l_ctuStride - 0];
        mask[0][2] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - l_ctuStride + 1];

        mask[1][0] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - 1];
        mask[1][1] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset - 0];
        mask[1][2] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset + 1];

        mask[2][0] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset + l_ctuStride - 1];
        mask[2][1] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset + l_ctuStride - 0];
        mask[2][2] = origHalfCTU[currRow*l_ctuStride + currCol + haloOffset + l_ctuStride + 1];

        // We only have to deal with the left and top edges. The bottom and right edges are never used as references
        // Put a zero on the samples of the mask that lie outside the frame
        if(halfCtuY+currRow==0){ // Top of frame
            mask[0][0] = 0;
            mask[0][1] = 0;
            mask[0][2] = 0;
            currScale = topScale;
            isTop = 1;
        }
        if(halfCtuY+currRow==frameHeight-1){ // Bottom of frame
            mask[2][0] = 0;
            mask[2][1] = 0;
            mask[2][2] = 0;
            currScale = bottomScale;
            isBottom = 1;
        }
        if(halfCtuX+currCol==0){ // Left of frame
            mask[0][0] = 0;
            mask[1][0] = 0;
            mask[2][0] = 0;
            currScale = leftScale;
            isLeft = 1;
        }
        if(halfCtuX+currCol==frameWidth-1){ // Right of frame
            mask[0][2] = 0;
            mask[1][2] = 0;
            mask[2][2] = 0;
            currScale = rightScale;
            isRight = 1;
        }

        // Check explicitly if the block is on the top-left or top-right corners to adjust the scale. The coefficients are already adjusted by entering the 2 conditionals
        unsigned int isTopLeft = (isTop && isLeft);
        unsigned int isTopRight = (isTop && isRight);
        unsigned int isBottomLeft = (isBottom && isLeft);
        unsigned int isBottomRight = (isBottom && isRight);
        currScale = select(currScale, topLeftScale, isTopLeft);
        currScale = select(currScale, topRightScale, isTopRight);
        currScale = select(currScale, bottomLeftScale, isBottomLeft);
        currScale = select(currScale, bottomRightScale, isBottomRight);

        result = 0;

        // if(halfCtuX==0 && halfCtuY==0 && currRow==1 && currCol==1)
        //     printf("Result %f\n", result);

        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                result += mask[i][j]*convKernel[i][j];
                // if(halfCtuX==0 && halfCtuY==0 && currRow==1 && currCol==1)
                //     printf("Result %f\n", result);
            }
        }

        result = round(result/currScale);

        filteredHalfCTU[currRow*128 + currCol] = result;

        currRow += rowsPerPass;
        currScale = scale;
        isTop = 0;
        isBottom = 0;
        isLeft = 0;
        isRight = 0;

    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      OFFLOAD FILTERED SAMPELS INTO GLOBAL MEMORY AGAIN
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int rowsRemaininig = min(64, frameHeight - halfCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    rowsPerPass = wgSize/128;
    haloOffset = 130+1;
    l_ctuStride = 130;


    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        // filteredFrame[g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128];
        filteredFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = filteredHalfCTU[pass*rowsPerPass*128 + (lid/128)*128 + lid%128];
    }


}

__kernel void filterFrame_2d_float_quarterCtu(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    float convKernel[3][3];
    
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            convKernel[i][j] = (float) convKernelLib[kernelIdx][i][j];        
        }
    }

    float fullScale = 0; for(int i=-1; i<=1; i++) for(int j=-1; j<=1; j++) fullScale+=convKernel[1+i][1+j];

    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/32.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*32;


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 samples around the input
    __local short origQuarterCTU[(128+1+1)*(32+1+1)];
    __local short filteredQuarterCTU[128*32];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*32/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;
    int haloOffset = 130+1;
    int l_ctuStride = 130;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        if(quarterCtuY + lid/128 + pass*rowsPerPass < frameHeight){
            origQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
        }
        else{
            origQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = -1;    
        }
        
    }

    // Fetch the halo

    int currRow, currCol;
   
    // HALO AT THE TOP AND BOTTOM
    currRow = lid/128;
    currRow = currRow*(32+1); // Either first or last row
    origQuarterCTU[1 + currRow*l_ctuStride + lid%128] = -1;

    if(( (g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) 
         //&& (quarterCtuY+currRow>0)
         ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128];
    }

    // One col of left and right edges: WIs in 0:1 fetch the first row of all columns, WIs 2:3 fetch the second row, and so on...
    currRow = lid/2;
    currCol = (lid%2) * 129; // Either first or last column
    if(lid<(1*2*32))
        origQuarterCTU[1*l_ctuStride + currRow*l_ctuStride + currCol] = -1;

    if( (lid<(1*2*32)) && 
        ((g_quarterCtuBaseIdx-1 + currRow*frameWidth + currCol)>0) &&
        ((g_quarterCtuBaseIdx-1 + currRow*frameWidth + currCol)<(frameWidth*frameHeight)) && 
        (quarterCtuX-1+currCol>0) &&
        (quarterCtuX-1+currCol<frameWidth-1) ){      
        // skip TL corner                                                              left neighbor col |
        origQuarterCTU[1*l_ctuStride + currRow*l_ctuStride + currCol] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-1 + currRow*frameWidth + currCol];
    }

    if(lid==0){
        origQuarterCTU[0]          = -1;
        origQuarterCTU[129]        = -1;
        origQuarterCTU[33*130]     = -1;
        origQuarterCTU[33*130+129] = -1;
    }

    // Top-left, top-right, bottom-left, bottom-right
    if(lid==0){
        // if((g_quarterCtuBaseIdx - frameWidth - 1)>0)
        if(quarterCtuY>0 && quarterCtuX>0)
            origQuarterCTU[0]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - frameWidth - 1]; // TL
        // if((g_quarterCtuBaseIdx - frameWidth + 128)>0)
        if(quarterCtuY>0 && quarterCtuX+128<frameWidth-1)
            origQuarterCTU[129]        = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - frameWidth + 128]; // TR
        // if((g_quarterCtuBaseIdx + 64*frameWidth -1)<(frameWidth*frameHeight))
        if(quarterCtuY+32<frameHeight-1 && quarterCtuX>0)
            origQuarterCTU[33*130]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth -1]; // BL
        // if((g_quarterCtuBaseIdx + 64*frameWidth + 128)<(frameWidth*frameHeight))
        if(quarterCtuY+32<frameHeight-1 && quarterCtuX+128<frameWidth-1)
            origQuarterCTU[33*130+129] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth + 128]; // BR
    }


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    int nPassesFilter = (128*32)/wgSize;

    float result;
    int mask[3][3];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    float currScale = fullScale;


    haloOffset = 130+1;
    l_ctuStride = 130;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){

        for(int dRow=-1; dRow<=1; dRow++){
            for(int dCol=-1; dCol<=1; dCol++){
                mask[1+dRow][1+dCol] = origQuarterCTU[haloOffset + currRow*l_ctuStride + currCol +  dRow*l_ctuStride + dCol];
                currScale = select(currScale, currScale-convKernel[1+dRow][1+dCol], mask[1+dRow][1+dCol]<0);
                mask[1+dRow][1+dCol] = select(mask[1+dRow][1+dCol], 0, mask[1+dRow][1+dCol]<0);
            }
        }

        result = 0;

        for(int i=0; i<3; i++){
            for(int j=0; j<3; j++){
                result += mask[i][j]*convKernel[i][j];
                // if(quarterCtuX+currCol==0 && quarterCtuY+currRow==31)
                //     printf("Result %f  || scale = %f\n", result, currScale);
            }
        }

        result = round(result/currScale);

        filteredQuarterCTU[currRow*128 + currCol] = result;

        currRow += rowsPerPass;
        currScale = fullScale;
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      OFFLOAD FILTERED SAMPELS INTO GLOBAL MEMORY AGAIN
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int rowsRemaininig = min(32, frameHeight - quarterCtuY); // Copy the whole quarter-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    rowsPerPass = wgSize/128;
    haloOffset = 130+1;
    l_ctuStride = 130;


    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        filteredFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = filteredQuarterCTU[pass*rowsPerPass*128 + (lid/128)*128 + lid%128];
    }


}

// This kernel is used to fetch the original samples and apply a low-pass filter
// The filtered samples are used as references during prediction
__kernel void filterFrame_1d_float(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    float convKernel[3];
    
    convKernel[0] = (float) convKernelLib[kernelIdx][0][0];
    convKernel[1] = (float) convKernelLib[kernelIdx][0][1];
    convKernel[2] = (float) convKernelLib[kernelIdx][0][2];

    float correctionFactor = 1/convKernel[0];

    convKernel[0] = 1;
    convKernel[1] = convKernel[1] * correctionFactor;
    convKernel[2] = 1;

    // Middle of the frame
    float fullScale = 4*convKernel[0] + 4*convKernel[1] + convKernel[1]*convKernel[1];
    // Corner of the frame: top-left, top-right, bottom-left, bottom-right
    float cornerScale = 1*convKernel[0] + 2*convKernel[1] + convKernel[1]*convKernel[1];
    // Edge but not corner: top, bottom, left, right
    float edgeScale = 2*convKernel[0] + 3*convKernel[1] + convKernel[1]*convKernel[1];

    int isTop=0, isBottom=0, isLeft=0, isRight=0;

    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/32.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*32;


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 sample around the input
    __local short origThenFinalQuarterCTU[130*34];
    __local float partialFilteredQuarterCTU[130*34];

    // // First fill with zeros, so that we don't have to correct the convKernels
    int currIdx, firstIdx = 0;
    while(firstIdx < 130*34){
        currIdx = firstIdx + lid;
        if(currIdx < 130*34){
            origThenFinalQuarterCTU[currIdx] = 0;
            partialFilteredQuarterCTU[currIdx] = 0;
        }
        firstIdx += wgSize;
    }

    // Fetch the inner region of the CTU, without the halo  
    int nPassesFetchOriginal = 128*32/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;
    int haloOffset = 130+1;
    int l_ctuStride = 130;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
    }

    // Fetch the halo
   
    // Upper and lower edges: WIs in 0:127 fetch the top, WIs 128:255 fetch the bottom
    int currRow = (lid/128)*(33); // Either first or last row of halo
   
    if(((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)>0) && ((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128];
    }
        
    // Left and right edges: WIs 0 and 1 fetch the first row of both columns, WIs 2 and 3 fetch the second row, and so on...
    currRow = 1 + lid/2; // 2 WIs per row. First and last rows were fetched earlier
    int currCol = lid%2; // Either first or last column
    if((lid<(2*32)) &&  // Necessary workitems to complete the task
        ((g_quarterCtuBaseIdx-1 + (currRow-1)*frameWidth + currCol*(l_ctuStride-1))>0) && // Avoid segfault on  __global
        ((g_quarterCtuBaseIdx-1 + (currRow-1)*frameWidth + currCol*(l_ctuStride-1))<(frameWidth*frameHeight)) && // Avoid segfault on  __global
        (quarterCtuX+currCol>0) && // Leftmost column of frame
        (quarterCtuX+currCol*(l_ctuStride-1)<(frameWidth-1))){ // Rightmost column of frame
        // skip TL corner                                                              left neighbor col |
        origThenFinalQuarterCTU[currRow*l_ctuStride + currCol*(l_ctuStride-1)] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-1 + (currRow-1)*frameWidth + currCol*(l_ctuStride-1)];
    }
        
    // Top-left, top-right, bottom-left, bottom-right
    if(lid==0){
        if( ((g_quarterCtuBaseIdx - frameWidth - 1)>0) && (quarterCtuX>0) && (quarterCtuY>0) )
            origThenFinalQuarterCTU[0]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - frameWidth - 1]; // TL
        if( ((g_quarterCtuBaseIdx - frameWidth + 128)>0) && (quarterCtuX+128<frameWidth-1) && (quarterCtuY>0) )
            origThenFinalQuarterCTU[129]        = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - frameWidth + 128]; // TR
        if( ((g_quarterCtuBaseIdx + 32*frameWidth -1)<(frameWidth*frameHeight)) && (quarterCtuX>0) && (quarterCtuY+32<frameHeight-1) )
            origThenFinalQuarterCTU[33*130]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth -1]; // BL
        if( ((g_quarterCtuBaseIdx + 32*frameWidth + 128)<(frameWidth*frameHeight)) && (quarterCtuX+128<frameWidth-1) && (quarterCtuY+32<frameHeight-1) )
            origThenFinalQuarterCTU[33*130+129] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth + 128]; // BR
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //                     1st - HORIZONTAL FILTER
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);


    // if(quarterCtuX==0 && quarterCtuY==1056 && lid==0){
    //     printf("quarterCtu @ XY = %dx%d\n", quarterCtuX, quarterCtuY);
    //     for(int i=16; i<34; i++){
    //         for(int j=0; j<130; j++){
    //             printf("%d,", origThenFinalQuarterCTU[i*130+j]);
    //         }
    //         printf("\n");
    //     }
    // }

    // return;

    int nPassesFilter = ((130-2)*34)/wgSize; // TOp and bottom halos must be filtered since they are used as references in the vertical operation Left and right halos do not ned filtering

    float result;
    float mask[3];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
  
    haloOffset = 1;
    l_ctuStride = 130;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){
        
        mask[0] = (float) origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset - 1];
        mask[1] = (float) origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset - 0];
        mask[2] = (float) origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset + 1];

        result = 0;

        for(int i=0; i<3; i++){
            result += mask[i]*convKernel[i];
        }

        // Horizontally filtered HalfCTU
        partialFilteredQuarterCTU[currRow*l_ctuStride + currCol + haloOffset] = result;

        currRow += rowsPerPass;
    }


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //                     2nd - VERTICAL FILTER
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);


    nPassesFilter = (128*32)/wgSize; // Now we dont have to worry about the halo

    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    float currScale = fullScale;

    haloOffset = 130 + 1; // Skip the top and bottom halos
    l_ctuStride = 130;
    // TODO: Use vload and dot-product operations

    for(int pass=0; pass<nPassesFilter; pass++){
        
        mask[0] = partialFilteredQuarterCTU[currRow*l_ctuStride + currCol + haloOffset - 1*l_ctuStride];
        mask[1] = partialFilteredQuarterCTU[currRow*l_ctuStride + currCol + haloOffset - 0*l_ctuStride];
        mask[2] = partialFilteredQuarterCTU[currRow*l_ctuStride + currCol + haloOffset + 1*l_ctuStride];


        // If bottom or top we must adjust the mask because it lies outside the frame
        if(quarterCtuY+currRow==0){ // Top of frame
            // mask[0] = 0;
            isTop = 1;
        }
        if(quarterCtuY+currRow==frameHeight-1){ // Bottom of frame
            // mask[2] = 0;
            isBottom = 1;
        }
        // If left or right the mask is completely inside the frame, but the scale must be adjusted because the horizontal filter does not correct it
        if(quarterCtuX+currCol==0){ // Left of frame
            isLeft = 1;
        }
        if(quarterCtuX+currCol==frameWidth-1){ // Right of frame
            isRight = 1;
        }

        // Check explicitly if the block is on the edges or corners to adjust the scale
        short isEdge = isLeft || isRight || isTop || isBottom;
        short isCorner = (isLeft + isRight + isTop + isBottom) >= 2; // If it is on two edges, it is a corner

        currScale = select(currScale, edgeScale, (int) isEdge);
        currScale = select(currScale, cornerScale, (int) isCorner);
        
        result = 0;

        for(int i=0; i<3; i++){
            result += mask[i]*convKernel[i];
        }

        result = round(result/currScale); // Rounded division
        
    
        // Horizontally filtered HalfCTU
        origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset] = (short) result;

        currRow += rowsPerPass;
        currScale = fullScale;
        isTop = 0;
        isBottom = 0;
        isLeft = 0;
        isRight = 0;

    }


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      OFFLOAD FILTERED SAMPLES INTO GLOBAL MEMORY AGAIN
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    barrier(CLK_LOCAL_MEM_FENCE);

    int rowsRemaininig = min(32, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;

    haloOffset = 130 + 1; // Skip the top and bottom halos
    l_ctuStride = 130;

    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        // filteredFrame[g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128];
        filteredFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + currRow*frameWidth + currCol] = origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset];

        currRow += rowsPerPass;
    }
}

__kernel void filterFrame_2d_float_5x5(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    float convKernel[5][5];
    
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            convKernel[i][j] = convKernelLib_5x5[kernelIdx][i][j];
        }
    }

    // float scale = 0; for(int i=0; i<5; i++) for(int j=0; j<5; j++) scale+=convKernel[i][j]; // convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];
    float fullScale = 0; for(int i=-2; i<=2; i++) for(int j=-2; j<=2; j++) fullScale+=convKernel[2+i][2+j]; // convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];

    int halfCtuColumns = ceil(frameWidth/128.0);
    int halfCtuRows = ceil(frameHeight/64.0);

    // int ctuIdx = wg; //wg;
    int halfCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int halfCtuX = (halfCtuIdx % halfCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int halfCtuY = (halfCtuIdx / halfCtuColumns)*64;


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 2 samples around the input
    __local short origHalfCTU[(128+2+2)*(64+2+2)];
    __local short filteredHalfCTU[128*64];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*64/wgSize;
    int rowsPerPass = wgSize/128;
    int g_halfCtuBaseIdx = halfCtuY*frameWidth + halfCtuX;
    int haloOffset = 132 + 132 + 2;
    int l_ctuStride = 132;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        if(halfCtuY + lid/128 + pass*rowsPerPass < frameHeight)
            origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
        else
            origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = -1;
    }

    // Fetch the halo
    int currRow, currCol;
    
    // Two rows of upper and lower edges: WIs in 0:127 fetch the TOP outer-edge, WIs in 128:255 fetch the TOP inner-edge
    //                                    WIs in 0:127 fetch the BOTTOM inner-edge, WIs in 128:255 fetch the BOTTOM outer-edge
    // HALO AT THE TOP
    currRow = select(0, 1, lid>=128);
    origHalfCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_halfCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_halfCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (halfCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origHalfCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    // HALO AT THE BOTTOM
    currRow = select(66, 67, lid>=128);
    origHalfCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_halfCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_halfCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (halfCtuY+currRow<frameHeight-1) ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origHalfCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }


    // Two cols of left and right edges: WIs in 0:3 fetch the first row of all columns, WIs 4:7 fetch the second row, and so on...
    currRow = lid/4;
    currCol = lid%4;
    currCol = select(currCol, currCol+128, currCol>=2); // Adjust cols 2 and 3 to be 130 and 131 (last 2 cols)
    origHalfCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = -1;
    if( (lid<(2*2*64)) && 
        ((g_halfCtuBaseIdx-2 + currRow*frameWidth + currCol)>0) &&
        ((g_halfCtuBaseIdx-2 + currRow*frameWidth + currCol)<(frameWidth*frameHeight)) && 
        (halfCtuX-2+currCol>0) &&
        (halfCtuX-2+currCol<frameWidth-1) ){      
        // skip TL corner                                                              left neighbor col |
        origHalfCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx-2 + currRow*frameWidth + currCol];
    }

    if(lid<16){
        origHalfCTU[cornerIdxLUT_5x5[lid]] = -1;
    }

    if(lid==0){
        // Top-Left corners and interfaces
        if(halfCtuX>0 && halfCtuY>0){
            origHalfCTU[0]          = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - 2*frameWidth - 2]; // Outer corner
            origHalfCTU[1]          = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - 2*frameWidth - 1]; // Top interface
            origHalfCTU[l_ctuStride] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - 1*frameWidth - 2]; // Left interface
            origHalfCTU[l_ctuStride+1] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - 1*frameWidth - 1]; // Inner corner
        }
                
        // Top-Right corners and interfaces
        if(halfCtuY>0){
            if(halfCtuX+128<frameWidth-1){
                origHalfCTU[l_ctuStride-2]          = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - 2*frameWidth + 128]; // Top interface
                origHalfCTU[2*l_ctuStride-2] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - 1*frameWidth + 128]; // Inner corner
            }
            if(halfCtuX+128+1<frameWidth-1){
                origHalfCTU[l_ctuStride-1]          = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - 2*frameWidth + 128+1]; // Outer corner
                origHalfCTU[2*l_ctuStride-1] = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx - 1*frameWidth + 128+1]; // Right interface
            }
        }
        
        // Bottom-Left corners and interfaces
        if(halfCtuX>0){
            if(halfCtuY+64<frameHeight-1){
                origHalfCTU[66*132]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 64*frameWidth -2]; // Left interface
                origHalfCTU[66*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 64*frameWidth -1]; // Inner corner
            }
            if(halfCtuY+65<frameHeight-1){
                origHalfCTU[67*132]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 65*frameWidth -2]; // Bottom interface
                origHalfCTU[67*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 65*frameWidth -1]; // Outer corner
            }
        }
        
        // Bottom-right corners and interfaces 
        if(halfCtuY+64<frameHeight-1 && halfCtuX+128+1<frameWidth-1) // Right interface
            origHalfCTU[67*l_ctuStride - 1]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 64*frameWidth + 128+1];
        if(halfCtuY+64<frameHeight-1 && halfCtuX+128<frameWidth-1) // Inner corner
            origHalfCTU[67*l_ctuStride - 2]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 64*frameWidth + 128];
        if(halfCtuY+64+1<frameHeight-1 && halfCtuX+128+1<frameWidth-1) // Outer corner
            origHalfCTU[68*l_ctuStride -1]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 65*frameWidth + 128+1];
        if(halfCtuY+64+1<frameHeight-1 && halfCtuX+128<frameWidth-1) // Bottom interface
            origHalfCTU[68*l_ctuStride -2]     = referenceFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + 65*frameWidth + 128];
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);


    int nPassesFilter = (128*64)/wgSize;

    float result;
    int mask[5][5];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    float currScale = fullScale;


    haloOffset = 132 + 132 + 2;
    l_ctuStride = 132;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){
        for(int dRow=-2; dRow<=2; dRow++){ // deltaRow and deltaCol to get the adjacent samples
            for(int dCol=-2; dCol<=2; dCol++){
                mask[2+dRow][2+dCol] = origHalfCTU[haloOffset + currRow*l_ctuStride + currCol + dRow*l_ctuStride + dCol];        
                // If sample is negative (invalid), we zero it and reduce the scale. Otherwise keep everything as usual
                currScale = select(currScale, currScale-convKernel[2+dRow][2+dCol], mask[2+dRow][2+dCol]<0);
                mask[2+dRow][2+dCol] = select(mask[2+dRow][2+dCol], 0, mask[2+dRow][2+dCol]<0);
            }
        }



        result = 0;

        for(int i=0; i<5; i++){
            for(int j=0; j<5; j++){
                result += mask[i][j]*convKernel[i][j];
                // if(halfCtuX==0 && halfCtuY==1024 && currRow==55 && currCol==0)
                //     printf("Result %f\n", result);
            }
        }

        result = round(result/currScale);

        filteredHalfCTU[currRow*128 + currCol] = result;

        currRow += rowsPerPass;
        currScale = fullScale;
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      OFFLOAD FILTERED SAMPELS INTO GLOBAL MEMORY AGAIN
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int rowsRemaininig = min(64, frameHeight - halfCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    rowsPerPass = wgSize/128;
    haloOffset = 130+1;
    l_ctuStride = 130;


    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        // filteredFrame[g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128];
        filteredFrame[rep*frameWidth*frameHeight + g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = filteredHalfCTU[pass*rowsPerPass*128 + (lid/128)*128 + lid%128];
    }


}

__kernel void filterFrame_2d_float_5x5_quarterCtu(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    float convKernel[5][5];
    
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            convKernel[i][j] = convKernelLib_5x5_float[kernelIdx][i][j];
        }
    }

    // float scale = 0; for(int i=0; i<5; i++) for(int j=0; j<5; j++) scale+=convKernel[i][j]; // convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];
    float fullScale = 0; for(int i=-2; i<=2; i++) for(int j=-2; j<=2; j++) fullScale+=convKernel[2+i][2+j]; // convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];

    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/32.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*32;


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 2 samples around the input
    __local short origQuarterCTU[(128+2+2)*(32+2+2)];
    __local short filteredQuarterCTU[128*32];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*32/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;

    // if(g_quarterCtuBaseIdx > frameWidth*frameHeight)
    //     return;

    int haloOffset = 132 + 132 + 2;
    int l_ctuStride = 132;
    int idx;

    // if(lid==0)
    //     printf("quarterCtu YX = %dx%d\n", quarterCtuY, quarterCtuX);

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        if(quarterCtuY + lid/128 + pass*rowsPerPass < frameHeight)
            origQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
        else
            origQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = -1;
    }

    // Fetch the halo
    int currRow, currCol;
    
    // Two rows of upper and lower edges: WIs in 0:127 fetch the TOP outer-edge, WIs in 128:255 fetch the TOP inner-edge
    //                                    WIs in 0:127 fetch the BOTTOM inner-edge, WIs in 128:255 fetch the BOTTOM outer-edge
    // HALO AT THE TOP
    currRow = select(0, 1, lid>=128);
    origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    // HALO AT THE BOTTOM
    currRow = select(34, 35, lid>=128);
    origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY+currRow<frameHeight-1) ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }

    // Two cols of left and right edges: WIs in 0:3 fetch the first row of all columns, WIs 4:7 fetch the second row, and so on...
    currRow = lid/4;
    currCol = lid%4;
    currCol = select(currCol, currCol+128, currCol>=2); // Adjust cols 2 and 3 to be 130 and 131 (last 2 cols)
    if(lid<(2*2*32))
        origQuarterCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = -1;
    if( (lid<(2*2*32)) && 
        ((g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol)>0) &&
        ((g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol)<(frameWidth*frameHeight)) && 
        (quarterCtuX-2+currCol>0) &&
        (quarterCtuX-2+currCol<frameWidth-1) ){      
        // skip TL corner                                                              left neighbor col |
        origQuarterCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol];
    }

    if(lid<16){
        origQuarterCTU[cornerIdxLUT_5x5_quarterCtu[lid]] = -1;
    }

    if(lid==0){
        // Top-Left corners and interfaces
        if(quarterCtuX>0 && quarterCtuY>0){
            origQuarterCTU[0]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 2*frameWidth - 2]; // Outer corner
            origQuarterCTU[1]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 2*frameWidth - 1]; // Top interface
            origQuarterCTU[l_ctuStride] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 1*frameWidth - 2]; // Left interface
            origQuarterCTU[l_ctuStride+1] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 1*frameWidth - 1]; // Inner corner
        }
                
        // Top-Right corners and interfaces
        if(quarterCtuY>0){
            if(quarterCtuX+128<frameWidth-1){
                origQuarterCTU[l_ctuStride-2]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 2*frameWidth + 128]; // Top interface
                origQuarterCTU[2*l_ctuStride-2] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 1*frameWidth + 128]; // Inner corner
            }
            if(quarterCtuX+128+1<frameWidth-1){
                origQuarterCTU[l_ctuStride-1]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 2*frameWidth + 128+1]; // Outer corner
                origQuarterCTU[2*l_ctuStride-1] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 1*frameWidth + 128+1]; // Right interface
            }
        }
        
        // Bottom-Left corners and interfaces
        if(quarterCtuX>0){
            if(quarterCtuY+32<frameHeight-1){
                origQuarterCTU[34*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth -2]; // Left interface
                origQuarterCTU[34*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth -1]; // Inner corner
            }
            if(quarterCtuY+33<frameHeight-1){
                origQuarterCTU[35*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 33*frameWidth -2]; // Bottom interface
                origQuarterCTU[35*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 33*frameWidth -1]; // Outer corner
            }
        }
        
        // Bottom-right corners and interfaces 
        if(quarterCtuY+32<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Right interface
            origQuarterCTU[35*l_ctuStride - 1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth + 128+1];
        if(quarterCtuY+32<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Inner corner
            origQuarterCTU[35*l_ctuStride - 2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth + 128];
        if(quarterCtuY+32+1<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Outer corner
            origQuarterCTU[36*l_ctuStride -1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 33*frameWidth + 128+1];
        if(quarterCtuY+32+1<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Bottom interface
            origQuarterCTU[36*l_ctuStride -2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 33*frameWidth + 128];
    
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);


    int nPassesFilter = (128*32)/wgSize;

    float result;
    int mask[5][5];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    float currScale = fullScale;


    haloOffset = 132 + 132 + 2;
    l_ctuStride = 132;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){
        for(int dRow=-2; dRow<=2; dRow++){ // deltaRow and deltaCol to get the adjacent samples
            for(int dCol=-2; dCol<=2; dCol++){
                mask[2+dRow][2+dCol] = origQuarterCTU[haloOffset + currRow*l_ctuStride + currCol + dRow*l_ctuStride + dCol];        
                // If sample is negative (invalid), we zero it and reduce the scale. Otherwise keep everything as usual
                currScale = select(currScale, currScale-convKernel[2+dRow][2+dCol], mask[2+dRow][2+dCol]<0);
                mask[2+dRow][2+dCol] = select(mask[2+dRow][2+dCol], 0, mask[2+dRow][2+dCol]<0);
            }
        }



        result = 0;

        for(int i=0; i<5; i++){
            for(int j=0; j<5; j++){
                result += mask[i][j]*convKernel[i][j];

            }
        }

        result = round(result/currScale);

        filteredQuarterCTU[currRow*128 + currCol] = result;

        currRow += rowsPerPass;
        currScale = fullScale;
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      OFFLOAD FILTERED SAMPELS INTO GLOBAL MEMORY AGAIN
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    barrier(CLK_LOCAL_MEM_FENCE);
    
    int rowsRemaininig = min(32, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    rowsPerPass = wgSize/128;
    haloOffset = 130+1;
    l_ctuStride = 130;


    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        // filteredFrame[g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128];
        filteredFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = filteredQuarterCTU[pass*rowsPerPass*128 + (lid/128)*128 + lid%128];
    }


}

__kernel void filterFrame_1d_float_5x5(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){   
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    float convKernel[5];

    convKernel[0] = (float) convKernelLib_5x5[kernelIdx][0][0];
    convKernel[1] = (float) convKernelLib_5x5[kernelIdx][0][1];
    convKernel[2] = (float) convKernelLib_5x5[kernelIdx][0][2];
    convKernel[3] = (float) convKernelLib_5x5[kernelIdx][0][4];
    convKernel[4] = (float) convKernelLib_5x5[kernelIdx][0][5];

    float fullScale = 0;
    for(int i=0; i<5; i++) 
        for(int j=0; j<5; j++)
            fullScale += convKernelLib_5x5[kernelIdx][i][j];

    float outerCornerScale = 0;
    for(int i=2; i<5; i++) 
        for(int j=2; j<5; j++)
            outerCornerScale += convKernelLib_5x5[kernelIdx][i][j];

    float innerCornerScale = 0;
    for(int i=1; i<5; i++) 
        for(int j=1; j<5; j++)
            innerCornerScale += convKernelLib_5x5[kernelIdx][i][j];

    float interfaceScale = 0;
    for(int i=1; i<5; i++) 
        for(int j=2; j<5; j++)
            interfaceScale += convKernelLib_5x5[kernelIdx][i][j];

    float outerEdgeScale = 0;
    for(int i=0; i<5; i++) 
        for(int j=2; j<5; j++)
            outerEdgeScale += convKernelLib_5x5[kernelIdx][i][j];

    float innerEdgeScale = 0;
    for(int i=0; i<5; i++) 
        for(int j=1; j<5; j++)
            innerEdgeScale += convKernelLib_5x5[kernelIdx][i][j];

    int isOuterTopBottom, isInnerTopBottom, isOuterLeftRight, isInnerLeftRight, isOuterEdge, isInnerEdge, isOuterCorner, isInnerCorner, isInterface;
    


    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/32.0);

    int quarterCtuIdx = wg;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*32;


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 sample around the input
    __local short origThenFinalQuarterCTU[132*36];
    __local float partialFilteredQuarterCTU[132*36];

    // // First fill with -1, so that we don't have to correct the convKernels
    int currIdx, firstIdx = 0;
    while(firstIdx < 132*36){
        currIdx = firstIdx + lid;
        if(currIdx < 132*36){
            origThenFinalQuarterCTU[currIdx] = -1;
            partialFilteredQuarterCTU[currIdx] = -1;
        }
        firstIdx += wgSize;
    }
    
    // Fetch the inner region of the CTU, without the halo  
    int nPassesFetchOriginal = 128*32/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;
    int haloOffset = 132 + 132 + 2;
    int l_ctuStride = 132;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        if(quarterCtuY + lid/128 + pass*rowsPerPass < frameHeight)
            origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
    }
    

    // Fetch the halo
    int currRow, currCol;
    
    // Two rows of upper and lower edges: WIs in 0:127 fetch the TOP outer-edge, WIs in 128:255 fetch the TOP inner-edge
    //                                    WIs in 0:127 fetch the BOTTOM inner-edge, WIs in 128:255 fetch the BOTTOM outer-edge
    // HALO AT THE TOP
    currRow = select(0, 1, lid>=128);
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    // HALO AT THE BOTTOM
    currRow = select(34, 35, lid>=128);
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY+currRow<frameHeight-1) ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }


    // Two cols of left and right edges: WIs in 0:3 fetch the first row of all columns, WIs 4:7 fetch the second row, and so on...
    currRow = lid/4;
    currCol = lid%4;
    currCol = select(currCol, currCol+128, currCol>=2); // Adjust cols 2 and 3 to be 130 and 131 (last 2 cols)
    if( (lid<(2*2*32)) && 
        ((g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol)>0) &&
        ((g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol)<(frameWidth*frameHeight)) && 
        (quarterCtuX-2+currCol>0) &&
        (quarterCtuX-2+currCol<frameWidth-1) ){      
        // skip TL corner                                                              left neighbor col |
        origThenFinalQuarterCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol];
    }



    if(lid==0){
        // Top-Left corners and interfaces
        if(quarterCtuX>0 && quarterCtuY>0){
            origThenFinalQuarterCTU[0]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 2*frameWidth - 2]; // Outer corner
            origThenFinalQuarterCTU[1]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 2*frameWidth - 1]; // Top interface
            origThenFinalQuarterCTU[l_ctuStride] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 1*frameWidth - 2]; // Left interface
            origThenFinalQuarterCTU[l_ctuStride+1] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 1*frameWidth - 1]; // Inner corner
        }
                
        // Top-Right corners and interfaces
        if(quarterCtuY>0){
            if(quarterCtuX+128<frameWidth-1){
                origThenFinalQuarterCTU[l_ctuStride-2]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 2*frameWidth + 128]; // Top interface
                origThenFinalQuarterCTU[2*l_ctuStride-2] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 1*frameWidth + 128]; // Inner corner
            }
            if(quarterCtuX+128+1<frameWidth-1){
                origThenFinalQuarterCTU[l_ctuStride-1]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 2*frameWidth + 128+1]; // Outer corner
                origThenFinalQuarterCTU[2*l_ctuStride-1] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - 1*frameWidth + 128+1]; // Right interface
            }
        }
        
        // Bottom-Left corners and interfaces
        if(quarterCtuX>0){
            if(quarterCtuY+32<frameHeight-1){
                origThenFinalQuarterCTU[34*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth -2]; // Left interface
                origThenFinalQuarterCTU[34*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth -1]; // Inner corner
            }
            if(quarterCtuY+33<frameHeight-1){
                origThenFinalQuarterCTU[35*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 33*frameWidth -2]; // Bottom interface
                origThenFinalQuarterCTU[35*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 33*frameWidth -1]; // Outer corner
            }
        }
        
        // Bottom-right corners and interfaces 
        if(quarterCtuY+32<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Right interface
            origThenFinalQuarterCTU[35*l_ctuStride - 1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth + 128+1];
        if(quarterCtuY+32<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Inner corner
            origThenFinalQuarterCTU[35*l_ctuStride - 2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 32*frameWidth + 128];
        if(quarterCtuY+32+1<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Outer corner
            origThenFinalQuarterCTU[36*l_ctuStride -1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 33*frameWidth + 128+1];
        if(quarterCtuY+32+1<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Bottom interface
            origThenFinalQuarterCTU[36*l_ctuStride -2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 33*frameWidth + 128];
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //                     1st - HORIZONTAL FILTER
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    int nPassesFilter = ((132-4)*36)/wgSize; // TOp and bottom halos must be filtered since they are used as references in the vertical operation Left and right halos do not ned filtering

    float result;
    float mask[5];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
  
    haloOffset = 2;
    l_ctuStride = 132;
    unsigned int isNeg = 0;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){
        if( (quarterCtuY+currRow>=2) && 
            (quarterCtuY+currRow-2<frameHeight) ){
            isNeg = 0;

            for(int d=-2; d<=2; d++){ // Delta from -2 to +2
                mask[2+d] = (float) origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset + d];
                isNeg = mask[2+d]<0;
                mask[2+d] = select(mask[2+d], (float) 0.0, isNeg);
            }

            result = 0;

            for(int i=0; i<5; i++){
                result += mask[i]*convKernel[i];
            }

            // Horizontally filtered quarterCTU
            partialFilteredQuarterCTU[currRow*l_ctuStride + currCol + haloOffset] = result;
        }
        currRow += rowsPerPass;
    }    


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //                     2nd - VERTICAL FILTER
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    nPassesFilter = (128*32)/wgSize; // Now we dont have to worry about the halo

    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    float currScale = fullScale;

    haloOffset = 132 + 132 + 2; // Skip the top and bottom halos
    l_ctuStride = 132;
    // TODO: Use vload and dot-product operations

    for(int pass=0; pass<nPassesFilter; pass++){
        
        isNeg = 0;

        for(int d=-2; d<=2; d++){ // Delta from -2 to +2
            mask[2+d] = partialFilteredQuarterCTU[currRow*l_ctuStride + currCol + haloOffset + d*l_ctuStride];
            isNeg = mask[2+d]<0;
            currScale = select(currScale, currScale-convKernel[2+d], isNeg);
            mask[2+d] = select(mask[2+d], (float) 0.0, isNeg);
        }

        result = 0;

        for(int i=0; i<5; i++){
            result += mask[i]*convKernel[i];
        }


        // int isOuterTopBottom, isInnerTopBottom, isOuterLeftRight, isInnerLeftRight, isOuterEdge, isInnerEdge, isOuterCorner, isInnerCorner, isInterface;

        isOuterTopBottom = select(0, 1, (quarterCtuY+currRow==0) || (quarterCtuY+currRow==frameHeight-1));
        isInnerTopBottom = select(0, 1, (quarterCtuY+currRow==1) || (quarterCtuY+currRow==frameHeight-2));
        isOuterLeftRight = select(0, 1, (quarterCtuX+currCol==0) || (quarterCtuX+currCol==frameWidth-1));
        isInnerLeftRight = select(0, 1, (quarterCtuX+currCol==1) || (quarterCtuX+currCol==frameWidth-2));

        isOuterCorner = isOuterTopBottom && isOuterLeftRight;
        isInnerCorner = isInnerTopBottom && isInnerLeftRight;

        isInterface = (isOuterLeftRight && isInnerTopBottom) || (isInnerLeftRight && isOuterTopBottom);

        isOuterEdge = !isOuterCorner && !isInterface && (isOuterTopBottom || isOuterLeftRight);
        isInnerEdge = !isInnerCorner && !isInterface && (isInnerTopBottom || isInnerLeftRight);
        
        currScale = select(currScale, outerCornerScale, isOuterCorner);
        currScale = select(currScale, innerCornerScale, isInnerCorner);
        currScale = select(currScale, outerEdgeScale, isOuterEdge);
        currScale = select(currScale, innerEdgeScale, isInnerEdge);
        currScale = select(currScale, interfaceScale, isInterface);
              
        result = round(result/currScale); // Rounded division
        
    // Horizontally filtered HalfCTU
        origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset] = (short) result;

        currRow += rowsPerPass;
        currScale = fullScale;
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      OFFLOAD FILTERED SAMPLES INTO GLOBAL MEMORY AGAIN
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=    

    barrier(CLK_LOCAL_MEM_FENCE);

    int rowsRemaininig = min(32, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;

    haloOffset = 132 + 132 + 2; // Skip the top and bottom halos
    l_ctuStride = 132;

    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        // filteredFrame[g_halfCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = origHalfCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128];
        filteredFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + currRow*frameWidth + currCol] = origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset];

        currRow += rowsPerPass;
    }

}