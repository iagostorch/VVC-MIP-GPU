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