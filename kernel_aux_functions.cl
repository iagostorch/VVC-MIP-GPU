#include "constants.cl"

int targetGid = -1;
int targetLid = 0;
int targetWg = -1;

// Based on MatrixIntraPrediction::predictionUpsampling1D
void upsamplePrediction_SizeId2(const short localPredBuffer[8*8], const int upsamplingHorizontal, const int upsamplingVertical, short *predictedBlock, const short refT[64], const short refL[64]){
    int cuWidth = 64;
    int cuHeight = 64;

    int reducedWidth = 8;
    int reducedHeight = 8;
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);




    // ######################################################################
    //      Variables shared for hotizontal and vertical interpolation
    int nPasses;

    int xPos, yPos, idx;
    int valueBefore, valueAfter;
    int isMiddle;
    int offsetInStride;

    // ####################################################
    //         Start with horizontal interpolation...
    // ####################################################

    int log2UpsamplingHorizontal = (int) log2((float) upsamplingHorizontal);
    int roundingOffsetHorizontal = 1 << (log2UpsamplingHorizontal - 1);
    // The block produced by horizontal interpolation has 64x8 samples
    nPasses = (reducedWidth*upsamplingHorizontal*reducedHeight)/wgSize;

    for(int pass=0; pass<nPasses; pass++){
        // Position each workitem will fill in final matrix
        idx = pass*wgSize + lid;
        yPos = idx/cuWidth;
        xPos = idx%cuWidth;
        isMiddle = xPos>=upsamplingHorizontal; // In this case, the left boundary is not used
        offsetInStride = xPos%upsamplingHorizontal+1; // Position inside one window where samples are being interpolated. BeforeReference has stride=0, first interpolated sample has stride=1

        // For the first couple of sample columns, the "before" reference is the refL buffer
        if(isMiddle == 0){
            valueBefore = refL[yPos*upsamplingHorizontal+upsamplingHorizontal-1];
            valueAfter = localPredBuffer[yPos*upsamplingHorizontal+(xPos>>log2UpsamplingHorizontal)]; // xPos>>3 is always zero when middle=0
        }
        else{
            valueBefore = localPredBuffer[yPos*upsamplingHorizontal+((xPos-upsamplingHorizontal)>>log2UpsamplingHorizontal)];
            valueAfter = localPredBuffer[yPos*upsamplingHorizontal+((xPos)>>log2UpsamplingHorizontal)];
        }
        
        int filteredSample = ((upsamplingHorizontal-offsetInStride)*valueBefore + offsetInStride*valueAfter + roundingOffsetHorizontal)>>log2UpsamplingHorizontal;
        
        if(wg==targetWg && lid==targetLid){
            printf("HORIZONTAL wg=%d lid=%d pass=%d (%dx%d) -> prev after: %d - %d, pred=%d, stride=%d, middle=%d\n", wg, lid, pass, xPos, yPos, valueBefore, valueAfter, filteredSample, offsetInStride, isMiddle);
        }

        // TODO: These samples will be necessary to conduct vertical interpoation. It is good to keep them in shared memory while the processing is not completely finished
        // If total shared memory is a problem, maybe keep these intermediate samples (64x8) in shared memory and copy them into global memory at the end
        predictedBlock[wg*cuWidth*cuHeight + (yPos*upsamplingHorizontal+upsamplingHorizontal-1)*cuWidth + xPos] = filteredSample;
    }


    // #######################################################
    //         Then continue to vertical interpolation...
    // #######################################################
    int log2UpsamplingVertical = (int) log2((float) upsamplingVertical);
    int roundingOffsetVertical = 1 << (log2UpsamplingVertical - 1);
    // The block produced by vertical interpolation has 64x64 samples
    nPasses = (cuWidth*cuHeight)/wgSize;
    
    for(int pass=0; pass<nPasses; pass++){
        // Position each workitem will fill in final matrix
        idx = pass*wgSize + lid;
        yPos = idx/cuWidth;
        xPos = idx%cuWidth;
        isMiddle = yPos>=upsamplingVertical; // In this case, the upper boundary is not used
        offsetInStride = yPos%upsamplingVertical+1; // Position inside one window where samples are being interpolated
        
        // For the first couple of sample rows, the "before" reference is the refT buffer    
        if(isMiddle == 0){
            valueBefore = refT[xPos];
            valueAfter = predictedBlock[wg*cuWidth*cuHeight + ((yPos>>log2UpsamplingVertical)+upsamplingVertical-1)*cuWidth + xPos];
        }
        else{
            // TODO: Shifting right and left with the same amount can be substituted by AND operation
            valueBefore = predictedBlock[wg*cuWidth*cuHeight + (((yPos>>log2UpsamplingVertical)<<log2UpsamplingVertical)-1)*cuWidth + xPos];
            valueAfter = predictedBlock[wg*cuWidth*cuHeight + (((yPos>>log2UpsamplingVertical)<<log2UpsamplingVertical)+upsamplingVertical-1)*cuWidth + xPos];
        }
        
        // Weighted average between references plus the rounding offset
        int filteredSample = ((upsamplingVertical-offsetInStride)*valueBefore + offsetInStride*valueAfter + roundingOffsetVertical)>>log2UpsamplingVertical;
        
        if(wg==targetWg && lid==targetLid){
            printf("VERTICAL wg=%d lid=%d pass=%d (%dx%d) -> prev after: %d - %d, pred=%d, stride=%d, middle=%d\n", wg, lid, pass, xPos, yPos, valueBefore, valueAfter, filteredSample, offsetInStride, isMiddle);
        }

        if(0 && wg==2 && lid==0){
            printf("WG=%d, lid=%d (%dx%d) Idx=%d\n", wg, lid, xPos, yPos, idx);
        }

        predictedBlock[wg*cuWidth*cuHeight + yPos*cuWidth + xPos] = filteredSample;
    }
}

// Each workitem computes the SAD at a set of locations, accumulate, and return to the kernel
long int sad_64x64(short *predictedBlock, short currentCuSamples[64*64]){
    int cuWidth = 64;
    int cuHeight = 64;

    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    int nPasses = (64*64)/wgSize;


    long int sad=0, temp=0;

    int idx, yPos, xPos;

    for(int pass=0; pass<nPasses; pass++){
        idx = pass*wgSize + lid;
        yPos = idx/cuWidth;
        xPos = idx%cuWidth;
        temp = abs(predictedBlock[wg*64*64 + yPos*64 + xPos] - currentCuSamples[yPos*64 + xPos]);
        sad += (long int) temp;
    }
    return sad;

}