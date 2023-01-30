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

    int boundaryStrideForCtu = ALL_TOTAL_CUS_PER_CTU*BOUNDARY_SIZE_Id2; // Each CU occupy BOUNDARY_SIZE_Id2 (=4) positions in the reduced boundaries buffers
    int currCtuBoundariesIdx = ctuIdx * boundaryStrideForCtu;

    // This buffer stores all predicted CUs inside the current CTU, with a single prediction mode
    // Each CU is processed by 64 workitems, where each workitem conducts the prediction of a single sample
    // When necessary, each workitem will process more than one CU
    // After all CUs are predicted with a single prediction mode, the buffer is moved into global memory and the next prediction mode is tested
    __local short reducedPredictedCtu[ALL_MAX_CUS_PER_CTU][(BOUNDARY_SIZE_Id2*2)*(BOUNDARY_SIZE_Id2*2)];

    __local short upsampledPredictedCtu[128*128]; // used to store the entire CTU after upsampling, before computing distortion

    int totalPredictionModes = PREDICTION_MODES_ID2 + TEST_TRANSPOSED_MODES*PREDICTION_MODES_ID2;
    // Each 64 workitems process one CU irrespective of the "original size" since the reduced prediciton is 64x64 for all of them
    const short itemsPerCuInPrediction = 64;
    const char sampleInCu = lid%itemsPerCuInPrediction; // Each workitem processes 1 sample
    int cusPerPass = wgSize/itemsPerCuInPrediction;
    int nPasses = nCusInCtu / cusPerPass;
    short cuIdxInCtu;
    int predSample;
    
    // Compute transposed index inside CU. Used for transposed MIP modes
    char transposedSampleInCu;
    char tempX, tempY;
    tempX = sampleInCu%REDUCED_PRED_SIZE_Id2;
    tempY = sampleInCu/REDUCED_PRED_SIZE_Id2;
    transposedSampleInCu = tempX*REDUCED_PRED_SIZE_Id2 + tempY;

    short8 reducedBoundaries, coefficients;
    short4 reducedT, reducedL;
 
    // TODO: Change this for loop to fetch the boundaries a single time for non-transposed and a single time for transposed
    // Use for(transp = [0,1]){ for(mode = [0,5])}
    for(int m=0; m<totalPredictionModes; m++){
        
        short mode = m%PREDICTION_MODES_ID2;
        short t = -(m/PREDICTION_MODES_ID2); // -1 because this value is used in select(), and select() only tests the MSB of the value
        short8 isTransp = (short8) (t,t,t,t,t,t,t,t);

        for(int pass=0; pass<nPasses; pass++){
            cuIdxInCtu = pass*cusPerPass + floor((float)lid/itemsPerCuInPrediction);

            reducedT = vload4((currCtuBoundariesIdx + ALL_stridedCusPerCtu[cuSizeIdx]*4 + cuIdxInCtu*BOUNDARY_SIZE_Id2)/4, unified_redT);
            reducedL = vload4((currCtuBoundariesIdx + ALL_stridedCusPerCtu[cuSizeIdx]*4 + cuIdxInCtu*BOUNDARY_SIZE_Id2)/4, unified_redL);

            // Create complete boundaries array based on transposed or not-transposed
            reducedBoundaries = select((short8)(reducedT, reducedL), (short8)(reducedL, reducedT), isTransp);

            short firstVal = reducedBoundaries.s0;
            // Apply inputOffset to all boundaries except the first, then zero the first. After this the boundaries are ready to be multiplied by the coefficients
            reducedBoundaries = reducedBoundaries - (short8) (firstVal); //(0, firstVal, firstVal, firstVal, firstVal, firstVal, firstVal, firstVal);

            int offset = reducedBoundaries.s0 + reducedBoundaries.s1 + reducedBoundaries.s2 + reducedBoundaries.s3 + reducedBoundaries.s4 + reducedBoundaries.s5 + reducedBoundaries.s6 + reducedBoundaries.s7;
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
            uchar8 vectorizedCoeffs = vload8(0, &mipMatrix16x16[mode][sampleInCu][0]);
            // Shift the coefficients to the right by 1 element, so that coeff 1 is in position [1]. Zero first coefficient beause it does not exist
            uint8 mask = (uint8)(0,0,1,2,3,4,5,6); 
            vectorizedCoeffs = shuffle(vectorizedCoeffs, mask); 
            vectorizedCoeffs.s0 = 0;
            // Dot function works with at most 4 values at a time. We must do the lower and higher part individually
            predSample  = offset;
            predSample += dot(convert_float4(reducedBoundaries.hi), convert_float4(vectorizedCoeffs.hi));
            predSample += dot(convert_float4(reducedBoundaries.lo), convert_float4(vectorizedCoeffs.lo));


            predSample = (predSample >> MIP_SHIFT_MATRIX) + firstVal;
            predSample = clamp(predSample, 0, (1<<10)-1);

            // Adjust the "correct" position inside the CU depending if the mode is transposed or not            
            short position = select((short) sampleInCu, (short) transposedSampleInCu, t);

            reducedPredictedCtu[cuIdxInCtu][position] = predSample;

            // Wait until all samples of the CTU are predicted because we will move it into global memory in sequence
            barrier(CLK_LOCAL_MEM_FENCE);

            /*    TRACE PREDICTION
            if(1 && cuIdxInCtu==0 && cuSizeIdx==ALL_AL_8x32 && mode==0 && ctuIdx==62 && lid==0){
                //for(int cu=0; cu<4; cu++){
                    printf("REDUCED PREDICTION: CTU %d, CU %d, Mode %d\n", ctuIdx, cuIdxInCtu, m);
                    // printf("SUB  Reduced boundaries: %d, %d, %d, %d, %d, %d, %d, %d\n", reducedBoundaries.s0, reducedBoundaries.s1, reducedBoundaries.s2, reducedBoundaries.s3, reducedBoundaries.s4, reducedBoundaries.s5, reducedBoundaries.s6, reducedBoundaries.s7);
                    // printf("Coeffs:\n");
                    // for(int i=0; i<64; i++){
                    //     printf("Sample: %d\n  ", i);
                    //     for(int j=0; j<7; j++){
                    //         printf("%d,", mipMatrix16x16[mode][i][j]);
                    //     }
                    //     printf("\n");
                    // }
                    for(int i=0; i<8; i++){
                        for(int j=0; j<8; j++){
                            printf("%d,", reducedPredictedCtu[cuIdxInCtu][i*8+j]);
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
        const int currCtuPredictionIdx = ctuIdx*ALL_TOTAL_CUS_PER_CTU*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
        int currCuPredictionIdx; // This will point to the start of the current CU in the reduced prediction global buffer (i.e., CTU position + offset)
        
        // Should we save the reduced prediction in global memory? If the upsampling is conducted in the same kernel it is not necessary
        int SAVE_REDUCED_PREDICTION_IN_GLOBAL = 1;
        
        if(SAVE_REDUCED_PREDICTION_IN_GLOBAL){
            for(int pass=0; pass<nPasses; pass++){
                cuIdxInCtu = pass*cusPerPass + floor((float)lid/itemsPerCuInPrediction);
                // Point to start of this CU size in global buffer
                currCuPredictionIdx = currCtuPredictionIdx + ALL_stridedCusPerCtu[cuSizeIdx]*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
                // Point to start of this CU specifically in global buffer
                currCuPredictionIdx += cuIdxInCtu*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
                // Point to start of the current mode in global buffer
                currCuPredictionIdx += m*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2;

                reducedPrediction[currCuPredictionIdx + sampleInCu] = reducedPredictedCtu[cuIdxInCtu][sampleInCu];
            }
            barrier(CLK_LOCAL_MEM_FENCE); // Wait until all predicted samples are moved. The next iteration overwrites the local prediction buffer
        }
    } // End of current mode
}

__kernel void upsampleDistortionSizeId2_ALL(__global short *reducedPrediction, const int frameWidth, const int frameHeight, __global long *SAD, __global long *SATD, __global short* originalSamples, __global short *unified_redT, __global short *unified_redL, __global short *unified_refT, __global short *unified_refL){
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

    int boundaryStrideForCtu = ALL_cusPerCtu[cuSizeIdx]*BOUNDARY_SIZE_Id2; // Each CU occupy BOUNDARY_SIZE_Id2 (=4) positions in the reduced boundaries buffers
    int currCtuBoundariesIdx = ctuIdx * boundaryStrideForCtu;

    const int upsamplingHorizontal = cuWidth / REDUCED_PRED_SIZE_Id2;
    const int upsamplingVertical = cuHeight / REDUCED_PRED_SIZE_Id2;

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

    __local int localSadEntireCtu[ALL_MAX_CUS_PER_CTU][PREDICTION_MODES_ID2*2];
    __local int localSatdEntireCtu[ALL_MAX_CUS_PER_CTU][PREDICTION_MODES_ID2*2];

    // This points to the current CTU in the global reduced prediction buffer
    const int currCtuPredictionIdx = ctuIdx*ALL_TOTAL_CUS_PER_CTU*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;

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
        for(int mode=0; mode<12; mode++){
            // -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            //
            //      FETCH THE REDUCED PREDICTION FOR CURRENT MODE AND CUs
    
            // Point to the start of current CTU
            idxCurrCuAndMode = ctuIdx*ALL_TOTAL_CUS_PER_CTU*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
            // Point to start of current CU size
            idxCurrCuAndMode += ALL_stridedCusPerCtu[cuSizeIdx]*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
            // Point to start of current CU
            idxCurrCuAndMode += currCu*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2*PREDICTION_MODES_ID2*2;
            // Point to start of current prediction mode
            idxCurrCuAndMode += mode*REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2;

            if(lid%itemsPerCuInUpsampling < (REDUCED_PRED_SIZE_Id2*REDUCED_PRED_SIZE_Id2)){
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
            int nPassesHorizontalUpsampling = max(1, (cuWidth*REDUCED_PRED_SIZE_Id2)/itemsPerCuInUpsampling);

            for(int pass=0; pass<nPassesHorizontalUpsampling; pass++){
                int idx = pass*itemsPerCuInUpsampling + lid%itemsPerCuInUpsampling;
                xPosInCu = idx%cuWidth;
                yPosInCu = (idx/cuWidth)*upsamplingVertical + upsamplingVertical-1;
                xPosInCtu = ALL_X_POS[cuSizeIdx][currCu]; // (currCu%cuColumnsPerCtu)*cuWidth + xPosInCu;
                yPosInCtu = ALL_Y_POS[cuSizeIdx][currCu]; // (currCu/cuColumnsPerCtu)*cuHeight + yPosInCu;

                isMiddle = xPosInCu>=upsamplingHorizontal; // In this case, the left boundary is not used
                offsetInStride = xPosInCu%upsamplingHorizontal+1; // Position inside one window where samples are being interpolated. BeforeReference has stride=0, first interpolated sample has stride=1            }

                if(lid%itemsPerCuInUpsampling < cuWidth*REDUCED_PRED_SIZE_Id2){ // For CUs 8x16 the horizontal upsampling is 1 and only 64 workitems work, the others are idle (in this case there is not upsampling, only copying)
                    // For the first couple of sample columns, the "before" reference is the refL buffer
                    if(isMiddle == 0){
                        // Predicted value that is before the current sample
                        valueBefore = refL[cuIdxInIteration*64 + yPosInCu];
                        // Predicted value that is after the current sample
                        valueAfter = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*REDUCED_PRED_SIZE_Id2 + (xPosInCu>>log2UpsamplingHorizontal)];
                    }
                    else{ // isMiddle == 1
                        valueBefore = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*REDUCED_PRED_SIZE_Id2 + (xPosInCu>>log2UpsamplingHorizontal) - 1];
                        valueAfter  = localReducedPrediction[cuIdxInIteration][(yPosInCu>>log2UpsamplingVertical)*REDUCED_PRED_SIZE_Id2 + (xPosInCu>>log2UpsamplingHorizontal)];
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
    if(lid < ALL_cusPerCtu[cuSizeIdx]*PREDICTION_MODES_ID2*2){
        int nPassesMoveSadIntoGlobal = max(1, ALL_cusPerCtu[cuSizeIdx]*PREDICTION_MODES_ID2*2);
        int idxInLocal, cu, mode, idxInGlobal;
        
        for(int pass=0; pass<nPassesMoveSadIntoGlobal; pass++){
            idxInLocal = pass*wgSize + lid;
            cu = idxInLocal / (PREDICTION_MODES_ID2*2);
            mode = idxInLocal % (PREDICTION_MODES_ID2*2);

            // Point to current CTU in global buffer
            idxInGlobal = ctuIdx*ALL_TOTAL_CUS_PER_CTU*PREDICTION_MODES_ID2*2;
            // Point to current CU size in global buffer
            idxInGlobal    += ALL_stridedCusPerCtu[cuSizeIdx]*PREDICTION_MODES_ID2*2;
            // Point to current CU and mode in global buffer
            idxInGlobal    += cu*PREDICTION_MODES_ID2*2 + mode;

            if(cu < ALL_cusPerCtu[cuSizeIdx]){
                SAD[idxInGlobal] = ( long ) localSadEntireCtu[cu][mode];
                SATD[idxInGlobal] = ( long ) localSatdEntireCtu[cu][mode];
            }
        }
    }
}