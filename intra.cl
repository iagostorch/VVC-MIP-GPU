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
    __local short int bufferGlobalRedT[ALL_MAX_CUS_PER_CTU*LARGEST_RED_BOUNDARY], bufferGlobalRedL[ALL_MAX_CUS_PER_CTU*LARGEST_RED_BOUNDARY];

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
        cuY = ALL_Y_POS[cuSizeIdx][cuIdx];
        cuX = ALL_X_POS[cuSizeIdx][cuIdx];
         
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

    //* TRACE REDUCED TOP BOUNDARIES FOR THE ENTIRE CTU
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
        int nPasses = max(1, (reducedBoundarySize*nCusInCtu)/wgSize); // 
        for(int pass=0; pass<nPasses; pass++){
            unified_redT[ctuIdx*ALL_TOTAL_CUS_PER_CTU*reducedBoundarySize + ALL_stridedCusPerCtu[cuSizeIdx]*reducedBoundarySize + pass*wgSize + lid] = bufferGlobalRedT[pass*wgSize + lid];
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
        cuY = ALL_Y_POS[cuSizeIdx][cuIdx];
        cuX = ALL_X_POS[cuSizeIdx][cuIdx];

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
        int nPasses = max(1, (reducedBoundarySize*nCusInCtu)/wgSize); // 
        for(int pass=0; pass<nPasses; pass++){
            unified_redL[ctuIdx*ALL_TOTAL_CUS_PER_CTU*reducedBoundarySize + ALL_stridedCusPerCtu[cuSizeIdx]*reducedBoundarySize + pass*wgSize + lid] = bufferGlobalRedL[pass*wgSize + lid];
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
