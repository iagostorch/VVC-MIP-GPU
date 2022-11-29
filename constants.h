enum CU_SIZE {
    _64x64 = 0,
    _32x32 = 1,
    _16x16 = 2,
    NUM_CU_SIZES = 3
};

#define TOTAL_CUS_PER_CTU 84

#define PREDICTION_MODES_ID2 6

#define REDUCED_PRED_SIZE_Id2 8

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

const unsigned char cuColumnsPerCtu[3] = {
    2,  // 64x64
    4,  // 32x32
    8   // 16x16
};

const unsigned char cuRowsPerCtu[3] = {
    2,  // 64x64
    4,  // 32x32
    8   // 16x16
};

// This is used as a stride when we must access information from multiple CU sizes in the same buffer
const unsigned char stridedCusPerCtu[4] = { 
    0,          // 64x64
    0+4,        // 32x32
    0+4+16,     // 16x16
    0+4+16+64   // TOTAL_CUS_PER_CTU
   };