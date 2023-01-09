// #include "constants.h"

#define REDUCED_PRED_SIZE_Id2 8
#define REDUCED_PRED_SIZE_Id1 4
#define REDUCED_PRED_SIZE_Id0 4

#define BOUNDARY_SIZE_Id2 4
#define BOUNDARY_SIZE_Id1 4
#define BOUNDARY_SIZE_Id0 2

#define MAX_CUS_PER_CTU 128

#define TOTAL_CUS_PER_CTU 532 // Sum of cusPerCtu for all supported CU sizes

#define PREDICTION_MODES_ID2 6
#define TEST_TRANSPOSED_MODES 1

__constant unsigned char MIP_SHIFT_MATRIX = 6;
__constant unsigned char MIP_OFFSET_MATRIX = 32;

enum CU_SIZE {
    _64x64 = 0,
    _32x32 = 1,
    _32x16 = 2,
    _16x32 = 3,
    _32x8 =  4,
    _8x32 =  5,
    _16x16 = 6,
    _16x8  = 7,
    _8x16  = 8,
    NUM_CU_SIZES = 9
};

const unsigned char widths[9] = {
                                        64,  // 64x64
                                        32,  // 32x32
                                        32,  // 32x16
                                        16,  // 16x32

                                        32,  // 32x8
                                        8,   // 8x32


                                        16,   // 16x16


                                        16,  // 16x8
                                        8,   // 8x16
};

const unsigned char heights[9] = {
                                        64,  // 64x64
                                        32,  // 32x32
                                        16,  // 32x16
                                        32,  // 16x32

                                        8,   // 32x8
                                        32,  // 8x32


                                        16,  // 16x16


                                        8,   // 16x8
                                        16   // 8x16
};

const unsigned char cusPerCtu[9] = {
    4,   // 64x64
    16,  // 32x32
    32,  // 32x16
    32,  // 16x32

    64,  // 32x8
    64,  // 8x32
    
    64,  // 16x16

    128, // 16x8
    128  // 8x16
};

// This is used as a stride when we must access information from multiple CU sizes in the same buffer
const unsigned short stridedCusPerCtu[10] = { 
    0,                          // 64x64
    0+4,                        // 32x32
    0+4+16,                     // 32x16
    0+4+16+32,                  // 16x32

    0+4+16+32+32,               // 32x8
    0+4+16+32+32+64,            // 8x32


    0+4+16+32+32+64+64,         // 16x16

    0+4+16+32+32+64+64+64,      // 16x8
    0+4+16+32+32+64+64+64+128,   // 8x16

    0+4+16+32+32+64+64+64+128+128   // TOTAL_CUS_PER_CTU
   };

// Used to access the boundaries of a specific CU size inside the unified buffer
const unsigned short stridedCompleteTopBoundaries[10] = {
  0,                                                                        // 64x64
  0 + 4*64,                                                                 // 32x32
  0 + 4*64 + 16*32,                                                         // 32x16
  0 + 4*64 + 16*32 + 32*32,                                                 // 16x32

  0 + 4*64 + 16*32 + 32*32 + 32*16,                                         // 32x8
  0 + 4*64 + 16*32 + 32*32 + 32*16 + 64*32,                                 // 8x32

  0 + 4*64 + 16*32 + 32*32 + 32*16 + 64*32 + 64*8,                          // 16x16

  0 + 4*64 + 16*32 + 32*32 + 32*16 + 64*32 + 64*8 + 64*16,                  // 16x8
  0 + 4*64 + 16*32 + 32*32 + 32*16 + 64*32 + 64*8 + 64*16 + 128*16,         // 8x16

  0 + 4*64 + 16*32 + 32*32 + 32*16 + 64*32 + 64*8 + 64*16 + 128*16 + 128*8  // TOTAL_TOP_BOUNDARIES_PER_CTU
};

const unsigned short stridedCompleteLeftBoundaries[10] = {
  0,                                                                        // 64x64
  0 + 4*64,                                                                 // 32x32
  0 + 4*64 + 16*32,                                                         // 32x16
  0 + 4*64 + 16*32 + 32*16,                                                 // 16x32

  0 + 4*64 + 16*32 + 32*16 + 32*32,                                         // 32x8
  0 + 4*64 + 16*32 + 32*16 + 32*32 + 64*8,                                  // 8x32

  0 + 4*64 + 16*32 + 32*16 + 32*32 + 64*8 + 64*32,                          // 16x16

  0 + 4*64 + 16*32 + 32*16 + 32*32 + 64*8 + 64*32 + 64*16,                  // 16x8
  0 + 4*64 + 16*32 + 32*16 + 32*32 + 64*8 + 64*32 + 64*16 + 128*8,          // 8x16

  0 + 4*64 + 16*32 + 32*16 + 32*32 + 64*8 + 64*32 + 64*16 + 128*8 + 128*16  // TOTAL_TOP_BOUNDARIES_PER_CTU
};