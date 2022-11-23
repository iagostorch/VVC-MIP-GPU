// #include "constants.h"

#define REDUCED_PRED_SIZE_Id2 8
#define REDUCED_PRED_SIZE_Id1 4
#define REDUCED_PRED_SIZE_Id0 4

#define BOUNDARY_SIZE_Id2 4
#define BOUNDARY_SIZE_Id1 4
#define BOUNDARY_SIZE_Id0 2

#define MAX_CUS_PER_CTU 64

#define TOTAL_CUS_PER_CTU 84 // Sum of cusPerCtu for all supported CU sizes

#define PREDICTION_MODES_ID2 6
#define TEST_TRANSPOSED_MODES 1

__constant unsigned char MIP_SHIFT_MATRIX = 6;
__constant unsigned char MIP_OFFSET_MATRIX = 32;

enum CU_SIZE {
    _64x64 = 0,
    _32x32 = 1,
    _16x16 = 2,
    NUM_CU_SIZES = 3
};

const unsigned char widths[3] = {
                                        64,
                                        32,
                                        16
};

const unsigned char heights[3] = {
                                        64,
                                        32,
                                        16
};

const unsigned char cusPerCtu[3] = {
    4,
    16,
    64
};

// This is used as a stride when we must access information from multiple CU sizes in the same buffer
const unsigned char stridedCusPerCtu[4] = { 
    0,          // 64x64
    0+4,        // 32x32
    0+4+16,     // 16x16
    0+4+16+64   // TOTAL_CUS_PER_CTU
   };