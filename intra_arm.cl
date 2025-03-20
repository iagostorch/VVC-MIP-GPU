#ifdef cl_intel_printf

#pragma OPENCL EXTENSION cl_intel_printf : enable

#pragma OPENCL EXTENSION cl_nv_compiler_options : enable

#endif

#include "mip_matrix.cl"
#include "kernel_aux_functions.cl"

#define BUFFER_SLOTS 2

__kernel void filterFrame_2d_int_quarterCtu_arm(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    int convKernel[3][3];
    
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            convKernel[i][j] = convKernelLib[kernelIdx][i][j];        
        }
    }

    int fullScale = 0; for(int i=-1; i<=1; i++) for(int j=-1; j<=1; j++) fullScale+=convKernel[1+i][1+j];

    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/16.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*16;

    // Esse return resolve o problema
    // Outside the frame
    if(quarterCtuY>=frameHeight)
        return;


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 samples around the input
    __local short origQuarterCTU[(128+1+1)*(16+1+1)];
    __local short filteredQuarterCTU[128*16];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*16/wgSize;
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
    // currRow = lid/128;
    currRow = 0;
    currRow = currRow*(16+1); // Either first or last row
    origQuarterCTU[1 + currRow*l_ctuStride + lid%128] = -1;

    if(( (g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) 
         //&& (quarterCtuY+currRow>0)
         ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128];
    }

    currRow = 1;
    currRow = currRow*(16+1); // Either first or last row
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
    if(lid<(1*2*16))
        origQuarterCTU[1*l_ctuStride + currRow*l_ctuStride + currCol] = -1;


    if( (lid<(1*2*16)) && 
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
        origQuarterCTU[17*130]     = -1;
        origQuarterCTU[17*130+129] = -1;
    }

    // Top-left, top-right, bottom-left, bottom-right
    if(lid==0){
        // if(wg>400) printf("WG %d -> Idx = %d\n", wg, rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - frameWidth - 1);
        // if((g_quarterCtuBaseIdx - frameWidth - 1)>0)
        if(quarterCtuY>0 && quarterCtuX>0)
            origQuarterCTU[0]          = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - frameWidth - 1]; // TL
        // if((g_quarterCtuBaseIdx - frameWidth + 128)>0)
        if(quarterCtuY>0 && quarterCtuX+128<frameWidth-1)
            origQuarterCTU[129]        = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx - frameWidth + 128]; // TR
        // if((g_quarterCtuBaseIdx + 64*frameWidth -1)<(frameWidth*frameHeight))
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX>0)
            origQuarterCTU[17*130]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -1]; // BL
        // if((g_quarterCtuBaseIdx + 64*frameWidth + 128)<(frameWidth*frameHeight))
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128<frameWidth-1)
            origQuarterCTU[17*130+129] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128]; // BR
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);
  
    int nPassesFilter = (128*16)/wgSize;

    int result;
    int mask[3][3];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    int currScale = fullScale;


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

        result = (result + currScale/2)/currScale;

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
    
    int rowsRemaininig = min(16, frameHeight - quarterCtuY); // Copy the whole quarter-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    rowsPerPass = wgSize/128;
    haloOffset = 130+1;
    l_ctuStride = 130;


    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        filteredFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = filteredQuarterCTU[pass*rowsPerPass*128 + (lid/128)*128 + lid%128];
    }

}

__kernel void filterFrame_1d_int_arm(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    int convKernel[3];
    
    convKernel[0] = convKernelLib[kernelIdx][0][0];
    convKernel[1] = convKernelLib[kernelIdx][0][1];
    convKernel[2] = convKernelLib[kernelIdx][0][2];

    // Middle of the frame
    int fullScale = 4*convKernel[0] + 4*convKernel[1] + convKernel[1]*convKernel[1];
    // Corner of the frame: top-left, top-right, bottom-left, bottom-right
    int cornerScale = 1*convKernel[0] + 2*convKernel[1] + convKernel[1]*convKernel[1];
    // Edge but not corner: top, bottom, left, right
    int edgeScale = 2*convKernel[0] + 3*convKernel[1] + convKernel[1]*convKernel[1];

    int isTop=0, isBottom=0, isLeft=0, isRight=0;

    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/16.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*16;

    // Esse return resolve o problema
    // Outside the frame
    if(quarterCtuY>=frameHeight)
        return;

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 sample around the input
    __local short origThenFinalQuarterCTU[130*18];
    __local unsigned int partialFilteredQuarterCTU[130*18];

    // // First fill with zeros, so that we don't have to correct the convKernels
    int currIdx, firstIdx = 0;
    while(firstIdx < 130*18){
        currIdx = firstIdx + lid;
        if(currIdx < 130*18){
            origThenFinalQuarterCTU[currIdx] = 0;
            partialFilteredQuarterCTU[currIdx] = 0;
        }
        firstIdx += wgSize;
    }

    // Fetch the inner region of the CTU, without the halo  
    int nPassesFetchOriginal = 128*16/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;
    int haloOffset = 130+1;
    int l_ctuStride = 130;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        if(quarterCtuY + lid/128 + pass*rowsPerPass < frameHeight){
            origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
        }            
        // else{
        //     origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = -1;
        // }
    }

    // Fetch the halo
   
    // Upper and lower edges: WIs in 0:127 fetch the top, WIs 128:255 fetch the bottom
    int currRow = 0; 
    if(((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)>0) && ((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128];
    }
    
    currRow = 17; 
    if(((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)>0) && ((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128];
    }
        
    // Left and right edges: WIs 0 and 1 fetch the first row of both columns, WIs 2 and 3 fetch the second row, and so on...
    currRow = 1 + lid/2; // 2 WIs per row. First and last rows were fetched earlier
    int currCol = lid%2; // Either first or last column
    if((lid<(2*16)) &&  // Necessary workitems to complete the task
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
        if( ((g_quarterCtuBaseIdx + 16*frameWidth -1)<(frameWidth*frameHeight)) && (quarterCtuX>0) && (quarterCtuY+16<frameHeight-1) )
            origThenFinalQuarterCTU[17*130]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -1]; // BL
        if( ((g_quarterCtuBaseIdx + 16*frameWidth + 128)<(frameWidth*frameHeight)) && (quarterCtuX+128<frameWidth-1) && (quarterCtuY+16<frameHeight-1) )
            origThenFinalQuarterCTU[17*130+129] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128]; // BR
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //                     1st - HORIZONTAL FILTER
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    int nPassesFilter = ((130-2)*18)/wgSize; // TOp and bottom halos must be filtered since they are used as references in the vertical operation Left and right halos do not ned filtering

    int result;
    int mask[3];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
  
    haloOffset = 1;
    l_ctuStride = 130;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){
        
        mask[0] = origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset - 1];
        mask[1] = origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset - 0];
        mask[2] = origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset + 1];

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

    nPassesFilter = (128*16)/wgSize; // Now we dont have to worry about the halo

    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    int currScale = fullScale;

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

        result = (result + currScale/2)/currScale; // Rounded division        
    
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

    int rowsRemaininig = min(16, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
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

__kernel void filterFrame_2d_float_quarterCtu_arm(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
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
    int quarterCtuRows = ceil(frameHeight/16.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*16;

    // Outside the frame
    if(quarterCtuY>=frameHeight)
        return;

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 samples around the input
    __local short origQuarterCTU[(128+1+1)*(16+1+1)];
    __local short filteredQuarterCTU[128*16];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*16/wgSize;
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
    // currRow = lid/128;
    currRow = 0;
    currRow = currRow*(16+1); // Either first or last row
    origQuarterCTU[1 + currRow*l_ctuStride + lid%128] = -1;

    if(( (g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) 
         //&& (quarterCtuY+currRow>0)
         ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-1*frameWidth + currRow*frameWidth + lid%128];
    }

    currRow = 1;
    currRow = currRow*(16+1); // Either first or last row
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
    if(lid<(1*2*16))
        origQuarterCTU[1*l_ctuStride + currRow*l_ctuStride + currCol] = -1;

    if( (lid<(1*2*16)) && 
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
        origQuarterCTU[17*130]     = -1;
        origQuarterCTU[17*130+129] = -1;
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
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX>0)
            origQuarterCTU[17*130]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -1]; // BL
        // if((g_quarterCtuBaseIdx + 64*frameWidth + 128)<(frameWidth*frameHeight))
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128<frameWidth-1)
            origQuarterCTU[17*130+129] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128]; // BR
    }


    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    int nPassesFilter = (128*16)/wgSize;

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
    
    int rowsRemaininig = min(16, frameHeight - quarterCtuY); // Copy the whole quarter-CTU or only the remaining rows when the CTU lies partially outside the frame
    int nPassesOffloadFiltered = 128*rowsRemaininig/wgSize;

    rowsPerPass = wgSize/128;
    haloOffset = 130+1;
    l_ctuStride = 130;


    // TODO: Increase vertical dimension of reference and filtered frame to avoid if-else in read and writes
    for(int pass=0; pass<nPassesOffloadFiltered; pass++){
        filteredFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128] = filteredQuarterCTU[pass*rowsPerPass*128 + (lid/128)*128 + lid%128];
    }


}

__kernel void filterFrame_1d_float_arm(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
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
    int quarterCtuRows = ceil(frameHeight/16.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*16;

    // Esse return resolve o problema
    // Outside the frame
    if(quarterCtuY>=frameHeight)
        return;

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 sample around the input
    __local short origThenFinalQuarterCTU[130*18];
    __local float partialFilteredQuarterCTU[130*18];

    // // First fill with zeros, so that we don't have to correct the convKernels
    int currIdx, firstIdx = 0;
    while(firstIdx < 130*18){
        currIdx = firstIdx + lid;
        if(currIdx < 130*18){
            origThenFinalQuarterCTU[currIdx] = 0;
            partialFilteredQuarterCTU[currIdx] = 0;
        }
        firstIdx += wgSize;
    }

    // Fetch the inner region of the CTU, without the halo  
    int nPassesFetchOriginal = 128*16/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;
    int haloOffset = 130+1;
    int l_ctuStride = 130;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        if(quarterCtuY + lid/128 + pass*rowsPerPass < frameHeight){
            origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
        }
        // else{
        //     origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = -1;
        // }
        
    }

    // Fetch the halo
   
    // Upper and lower edges: WIs in 0:127 fetch the top, WIs 128:255 fetch the bottom
    int currRow = 0;  // Either first or last row of halo
    if(((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)>0) && ((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 17;  // Either first or last row of halo
    if(((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)>0) && ((g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[1 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-frameWidth + currRow*frameWidth + lid%128];
    }
        
    // Left and right edges: WIs 0 and 1 fetch the first row of both columns, WIs 2 and 3 fetch the second row, and so on...
    currRow = 1 + lid/2; // 2 WIs per row. First and last rows were fetched earlier
    int currCol = lid%2; // Either first or last column
    if((lid<(2*16)) &&  // Necessary workitems to complete the task
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
        if( ((g_quarterCtuBaseIdx + 16*frameWidth -1)<(frameWidth*frameHeight)) && (quarterCtuX>0) && (quarterCtuY+16<frameHeight-1) )
            origThenFinalQuarterCTU[17*130]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -1]; // BL
        if( ((g_quarterCtuBaseIdx + 16*frameWidth + 128)<(frameWidth*frameHeight)) && (quarterCtuX+128<frameWidth-1) && (quarterCtuY+16<frameHeight-1) )
            origThenFinalQuarterCTU[17*130+129] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128]; // BR
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //                     1st - HORIZONTAL FILTER
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    int nPassesFilter = ((130-2)*18)/wgSize; // TOp and bottom halos must be filtered since they are used as references in the vertical operation Left and right halos do not ned filtering

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


    nPassesFilter = (128*16)/wgSize; // Now we dont have to worry about the halo

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

    int rowsRemaininig = min(16, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
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

__kernel void filterFrame_2d_int_5x5_quarterCtu_arm(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    int convKernel[5][5];
    
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            convKernel[i][j] = convKernelLib_5x5[kernelIdx][i][j];
        }
    }

    // float scale = 0; for(int i=0; i<5; i++) for(int j=0; j<5; j++) scale+=convKernel[i][j]; // convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];
    int fullScale = 0; for(int i=-2; i<=2; i++) for(int j=-2; j<=2; j++) fullScale+=convKernel[2+i][2+j]; // convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];

    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/16.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*16;

    if(quarterCtuY>=frameHeight)
        return;

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 2 samples around the input
    __local short origQuarterCTU[(128+2+2)*(16+2+2)];
    __local short filteredQuarterCTU[128*16];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*16/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;

    // if(g_quarterCtuBaseIdx > frameWidth*frameHeight)
    //     return;

    int haloOffset = 132 + 132 + 2;
    int l_ctuStride = 132;
    int idx;

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
    currRow = 0; 
    origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 1; 
    origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    // HALO AT THE BOTTOM
    currRow = 18;
    origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY+currRow<frameHeight-1) ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 19;
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
    if(lid<(2*2*16))
        origQuarterCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = -1;
    if( (lid<(2*2*16)) && 
        ((g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol)>0) &&
        ((g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol)<(frameWidth*frameHeight)) && 
        (quarterCtuX-2+currCol>0) &&
        (quarterCtuX-2+currCol<frameWidth-1) ){      
        // skip TL corner                                                              left neighbor col |
        origQuarterCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol];
    }

    if(lid<16){
        origQuarterCTU[cornerIdxLUT_5x5_quarterCtu_arm[lid]] = -1;
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
            if(quarterCtuY+16<frameHeight-1){
                origQuarterCTU[18*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -2]; // Left interface
                origQuarterCTU[18*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -1]; // Inner corner
            }
            if(quarterCtuY+17<frameHeight-1){
                origQuarterCTU[19*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth -2]; // Bottom interface
                origQuarterCTU[19*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth -1]; // Outer corner
            }
        }
        
        // Bottom-right corners and interfaces 
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Right interface
            origQuarterCTU[19*l_ctuStride - 1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128+1];
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Inner corner
            origQuarterCTU[19*l_ctuStride - 2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128];
        if(quarterCtuY+16+1<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Outer corner
            origQuarterCTU[20*l_ctuStride -1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth + 128+1];
        if(quarterCtuY+16+1<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Bottom interface
            origQuarterCTU[20*l_ctuStride -2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth + 128];
    
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);  

    int nPassesFilter = (128*16)/wgSize;

    int result;
    int mask[5][5];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    int currScale = fullScale;


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

        result = (result + currScale/2)/currScale;

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
    
    int rowsRemaininig = min(16, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
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


__kernel void filterFrame_1d_int_5x5_arm(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){   
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    int convKernel[5];

    convKernel[0] = convKernelLib_5x5[kernelIdx][0][0];
    convKernel[1] = convKernelLib_5x5[kernelIdx][0][1];
    convKernel[2] = convKernelLib_5x5[kernelIdx][0][2];
    convKernel[3] = convKernelLib_5x5[kernelIdx][0][3];
    convKernel[4] = convKernelLib_5x5[kernelIdx][0][4];

    int fullScale = 0;
    for(int i=0; i<5; i++) 
        for(int j=0; j<5; j++)
            fullScale += convKernelLib_5x5[kernelIdx][i][j];

    int outerCornerScale = 0;
    for(int i=2; i<5; i++) 
        for(int j=2; j<5; j++)
            outerCornerScale += convKernelLib_5x5[kernelIdx][i][j];

    int innerCornerScale = 0;
    for(int i=1; i<5; i++) 
        for(int j=1; j<5; j++)
            innerCornerScale += convKernelLib_5x5[kernelIdx][i][j];

    int interfaceScale = 0;
    for(int i=1; i<5; i++) 
        for(int j=2; j<5; j++)
            interfaceScale += convKernelLib_5x5[kernelIdx][i][j];

    int outerEdgeScale = 0;
    for(int i=0; i<5; i++) 
        for(int j=2; j<5; j++)
            outerEdgeScale += convKernelLib_5x5[kernelIdx][i][j];

    int innerEdgeScale = 0;
    for(int i=0; i<5; i++) 
        for(int j=1; j<5; j++)
            innerEdgeScale += convKernelLib_5x5[kernelIdx][i][j];

    int isOuterTopBottom, isInnerTopBottom, isOuterLeftRight, isInnerLeftRight, isOuterEdge, isInnerEdge, isOuterCorner, isInnerCorner, isInterface;
    


    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/16.0);

    int quarterCtuIdx = wg;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*16;

    // Outside the frame
    if(quarterCtuY>=frameHeight)
        return;

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 sample around the input
    __local short origThenFinalQuarterCTU[132*20];
    __local unsigned int partialFilteredQuarterCTU[132*20];

    // // First fill with -1, so that we don't have to correct the convKernels
    int currIdx, firstIdx = 0;
    while(firstIdx < 132*20){
        currIdx = firstIdx + lid;
        if(currIdx < 132*20){
            origThenFinalQuarterCTU[currIdx] = -1;
            partialFilteredQuarterCTU[currIdx] = -1;
        }
        firstIdx += wgSize;
    }
    
    // Fetch the inner region of the CTU, without the halo  
    int nPassesFetchOriginal = 128*16/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;
    int haloOffset = 132 + 132 + 2;
    int l_ctuStride = 132;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        if(quarterCtuY + lid/128 + pass*rowsPerPass < frameHeight){
            origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
        }
        else{
            origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = -1;
        }
    }
    

    // Fetch the halo
    int currRow, currCol;
    
    // Two rows of upper and lower edges: WIs in 0:127 fetch the TOP outer-edge, WIs in 128:255 fetch the TOP inner-edge
    //                                    WIs in 0:127 fetch the BOTTOM inner-edge, WIs in 128:255 fetch the BOTTOM outer-edge
    // HALO AT THE TOP
    currRow = 0; 
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 1; 
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    // HALO AT THE BOTTOM
    currRow = 18; 
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY+currRow<frameHeight-1) ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 19; 
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
    if( (lid<(2*2*16)) && 
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
            if(quarterCtuY+16<frameHeight-1){
                origThenFinalQuarterCTU[18*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -2]; // Left interface
                origThenFinalQuarterCTU[18*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -1]; // Inner corner
            }
            if(quarterCtuY+17<frameHeight-1){
                origThenFinalQuarterCTU[19*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth -2]; // Bottom interface
                origThenFinalQuarterCTU[19*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth -1]; // Outer corner
            }
        }
        
        // Bottom-right corners and interfaces 
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Right interface
            origThenFinalQuarterCTU[19*l_ctuStride - 1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128+1];
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Inner corner
            origThenFinalQuarterCTU[19*l_ctuStride - 2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128];
        if(quarterCtuY+16+1<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Outer corner
            origThenFinalQuarterCTU[20*l_ctuStride -1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth + 128+1];
        if(quarterCtuY+16+1<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Bottom interface
            origThenFinalQuarterCTU[20*l_ctuStride -2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth + 128];
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //                     1st - HORIZONTAL FILTER
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    int nPassesFilter = ((132-4)*20)/wgSize; // TOp and bottom halos must be filtered since they are used as references in the vertical operation Left and right halos do not ned filtering

    int result;
    int mask[5];
    
    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
  
    haloOffset = 2;
    l_ctuStride = 132;
    int isNeg = 0;
    // TODO: Use vload and dot-product operations
    for(int pass=0; pass<nPassesFilter; pass++){
        if( (quarterCtuY+currRow>=2) && 
            (quarterCtuY+currRow-2<frameHeight) ){
            isNeg = 0;

            for(int d=-2; d<=2; d++){ // Delta from -2 to +2
                mask[2+d] = origThenFinalQuarterCTU[currRow*l_ctuStride + currCol + haloOffset + d];
                isNeg = mask[2+d]<0;
                mask[2+d] = select(mask[2+d], 0, isNeg);
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

    nPassesFilter = (128*16)/wgSize; // Now we dont have to worry about the halo

    currRow = lid/128;
    currCol = lid%128;
    rowsPerPass = wgSize/128;
    int currScale = fullScale;

    haloOffset = 132 + 132 + 2; // Skip the top and bottom halos
    l_ctuStride = 132;
    // TODO: Use vload and dot-product operations

    for(int pass=0; pass<nPassesFilter; pass++){
        
        isNeg = 0;

        for(int d=-2; d<=2; d++){ // Delta from -2 to +2
            mask[2+d] = partialFilteredQuarterCTU[currRow*l_ctuStride + currCol + haloOffset + d*l_ctuStride];
            isNeg = mask[2+d]<0;
            currScale = select(currScale, currScale-convKernel[2+d], isNeg);
            mask[2+d] = select(mask[2+d], 0, isNeg);
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
              
        result = (result + currScale/2) / currScale;
        
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

    int rowsRemaininig = min(16, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
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

__kernel void filterFrame_2d_float_5x5_quarterCtu_arm(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    float convKernel[5][5];
    
    for(int i=0; i<5; i++){
        for(int j=0; j<5; j++){
            // convKernel[i][j] = convKernelLib_5x5_float[kernelIdx][i][j];
            convKernel[i][j] = (float) convKernelLib_5x5[kernelIdx][i][j];
        }
    }

    // float scale = 0; for(int i=0; i<5; i++) for(int j=0; j<5; j++) scale+=convKernel[i][j]; // convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];
    float fullScale = 0; for(int i=-2; i<=2; i++) for(int j=-2; j<=2; j++) fullScale+=convKernel[2+i][2+j]; // convKernel[0][0]+convKernel[0][1]+convKernel[0][2]+convKernel[1][0]+convKernel[1][1]+convKernel[1][2]+convKernel[2][0]+convKernel[2][1]+convKernel[2][2];

    int quarterCtuColumns = ceil(frameWidth/128.0);
    int quarterCtuRows = ceil(frameHeight/16.0);

    // int ctuIdx = wg; //wg;
    int quarterCtuIdx = wg;
    // int ctuX = (ctuIdx % halfCtuColumns)*128;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    // int ctuY = (ctuIdx / halfCtuColumns)*64;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*16;

    // Esse return resolve o problema
    // Outside the frame
    if(quarterCtuY>=frameHeight)
        return;

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 2 samples around the input
    __local short origQuarterCTU[(128+2+2)*(16+2+2)];
    __local short filteredQuarterCTU[128*16];


    // Fetch the inner region of the CTU, without the halo  

    int nPassesFetchOriginal = 128*16/wgSize;
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
    currRow = 0;
    origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 1;
    origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    // HALO AT THE BOTTOM
    currRow = 18; 
    origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = -1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY+currRow<frameHeight-1) ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 19; 
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
    if(lid<(2*2*16))
        origQuarterCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = -1;
    if( (lid<(2*2*16)) && 
        ((g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol)>0) &&
        ((g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol)<(frameWidth*frameHeight)) && 
        (quarterCtuX-2+currCol>0) &&
        (quarterCtuX-2+currCol<frameWidth-1) ){      
        // skip TL corner                                                              left neighbor col |
        origQuarterCTU[2*l_ctuStride + currRow*l_ctuStride + currCol] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2 + currRow*frameWidth + currCol];
    }

    if(lid<16){
        origQuarterCTU[cornerIdxLUT_5x5_quarterCtu_arm[lid]] = -1;
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
            if(quarterCtuY+16<frameHeight-1){
                origQuarterCTU[18*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -2]; // Left interface
                origQuarterCTU[18*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -1]; // Inner corner
            }
            if(quarterCtuY+17<frameHeight-1){
                origQuarterCTU[19*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth -2]; // Bottom interface
                origQuarterCTU[19*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth -1]; // Outer corner
            }
        }
        
        // Bottom-right corners and interfaces 
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Right interface
            origQuarterCTU[19*l_ctuStride - 1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128+1];
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Inner corner
            origQuarterCTU[19*l_ctuStride - 2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128];
        if(quarterCtuY+16+1<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Outer corner
            origQuarterCTU[20*l_ctuStride -1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth + 128+1];
        if(quarterCtuY+16+1<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Bottom interface
            origQuarterCTU[20*l_ctuStride -2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth + 128];
    
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    int nPassesFilter = (128*16)/wgSize;

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
    
    int rowsRemaininig = min(16, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
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

__kernel void filterFrame_1d_float_5x5_arm(__global short *referenceFrame, __global short *filteredFrame, const int frameWidth, const int frameHeight, const int kernelIdx, const int rep){   
    
    int gid = get_global_id(0);
    int wg = get_group_id(0);
    int lid = get_local_id(0);
    int wgSize = get_local_size(0);

    float convKernel[5];

    convKernel[0] = (float) convKernelLib_5x5[kernelIdx][0][0];
    convKernel[1] = (float) convKernelLib_5x5[kernelIdx][0][1];
    convKernel[2] = (float) convKernelLib_5x5[kernelIdx][0][2];
    convKernel[3] = (float) convKernelLib_5x5[kernelIdx][0][3];
    convKernel[4] = (float) convKernelLib_5x5[kernelIdx][0][4];

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
    int quarterCtuRows = ceil(frameHeight/16.0);

    int quarterCtuIdx = wg;
    int quarterCtuX = (quarterCtuIdx % quarterCtuColumns)*128;
    int quarterCtuY = (quarterCtuIdx / quarterCtuColumns)*16;

    // Esse return resolve o problema
    // Outside the frame
    if(quarterCtuY>=frameHeight)
        return;

    // if(quarterCtuY>0)
    //     return;

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //
    //      FETCH THE ORIGINAL SAMPLES FROM __global INTO __local MEMORY
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    
    // Each WG processes one CTU. It requires a halo of 1 sample around the input
    __local short origThenFinalQuarterCTU[132*20];
    __local float partialFilteredQuarterCTU[132*20];

    // // First fill with -1, so that we don't have to correct the convKernels
    int currIdx, firstIdx = 0;
    while(firstIdx < 132*20){
        currIdx = firstIdx + lid;
        if(currIdx < 132*20){
            origThenFinalQuarterCTU[currIdx] = -1;
            partialFilteredQuarterCTU[currIdx] = -1;
        }
        firstIdx += wgSize;
    }
    
    // Fetch the inner region of the CTU, without the halo  
    int nPassesFetchOriginal = 128*16/wgSize;
    int rowsPerPass = wgSize/128;
    int g_quarterCtuBaseIdx = quarterCtuY*frameWidth + quarterCtuX;
    int haloOffset = 132 + 132 + 2;
    int l_ctuStride = 132;
    int idx;

    for(int pass=0; pass<nPassesFetchOriginal; pass++){
        if(quarterCtuY + lid/128 + pass*rowsPerPass < frameHeight){
            origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + pass*rowsPerPass*frameWidth + (lid/128)*frameWidth + lid%128];
        }
        else{
            origThenFinalQuarterCTU[haloOffset + pass*rowsPerPass*l_ctuStride + (lid/128)*l_ctuStride + lid%128] = -1;
        }
            
    }
    

    // Fetch the halo
    int currRow, currCol;
    
    // Two rows of upper and lower edges: WIs in 0:127 fetch the TOP outer-edge, WIs in 128:255 fetch the TOP inner-edge
    //                                    WIs in 0:127 fetch the BOTTOM inner-edge, WIs in 128:255 fetch the BOTTOM outer-edge
    // HALO AT THE TOP
    currRow = 0;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 1;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY>0)){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    // HALO AT THE BOTTOM
    currRow = 18;
    if(( (g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)>0) && 
         ((g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128)<frameWidth*frameHeight) && 
         (quarterCtuY+currRow<frameHeight-1) ){
        // skip 1st col (corner)                        point to 1st row, 2nd col of halo |
        origThenFinalQuarterCTU[2 + currRow*l_ctuStride + lid%128] = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx-2*frameWidth + currRow*frameWidth + lid%128];
    }
    currRow = 19;
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
    if( (lid<(2*2*16)) && 
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
            if(quarterCtuY+16<frameHeight-1){
                origThenFinalQuarterCTU[18*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -2]; // Left interface
                origThenFinalQuarterCTU[18*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth -1]; // Inner corner
            }
            if(quarterCtuY+17<frameHeight-1){
                origThenFinalQuarterCTU[19*132]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth -2]; // Bottom interface
                origThenFinalQuarterCTU[19*132+1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth -1]; // Outer corner
            }
        }
        
        // Bottom-right corners and interfaces 
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Right interface
            origThenFinalQuarterCTU[19*l_ctuStride - 1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128+1];
        if(quarterCtuY+16<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Inner corner
            origThenFinalQuarterCTU[19*l_ctuStride - 2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 16*frameWidth + 128];
        if(quarterCtuY+16+1<frameHeight-1 && quarterCtuX+128+1<frameWidth-1) // Outer corner
            origThenFinalQuarterCTU[20*l_ctuStride -1]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth + 128+1];
        if(quarterCtuY+16+1<frameHeight-1 && quarterCtuX+128<frameWidth-1) // Bottom interface
            origThenFinalQuarterCTU[20*l_ctuStride -2]     = referenceFrame[rep*frameWidth*frameHeight + g_quarterCtuBaseIdx + 17*frameWidth + 128];
    }

    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
    //                     1st - HORIZONTAL FILTER
    //
    //      FILTER THE SAMPLES IN LOCAL MEMORY AND SAVE INTO ANOTHER LOCAL BUFFER
    //
    //  -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    barrier(CLK_LOCAL_MEM_FENCE);

    int nPassesFilter = ((132-4)*20)/wgSize; // TOp and bottom halos must be filtered since they are used as references in the vertical operation Left and right halos do not ned filtering

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

    nPassesFilter = (128*16)/wgSize; // Now we dont have to worry about the halo

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

    int rowsRemaininig = min(16, frameHeight - quarterCtuY); // Copy the whole half-CTU or only the remaining rows when the CTU lies partially outside the frame
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


