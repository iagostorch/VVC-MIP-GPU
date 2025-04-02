
# VVC-MIP-GPU

A GPU based engine for the Matrix-based Intra Prediction (MIP) procedure of the Versatile Video Coding (VVC) standard. It is a GPU-fiendly version of the MIP modes bundled into the VTM encoder, implemented in OpenCL to seize the performance of GPUs from most manufacturers.


## Building the project

This project is based on C++ and OpenCL, and it has dependencies with libboost. A convenient makefile is supplied for building the project locally.



```bash
  cd VVC-Affine-GPU
  make
```
    
## Usage/Examples

The directory ```data/``` contains examples of input files for 1080p videos (1920x1080 samples). The execution requires one .csv file containing the original frame samples (i.e., the frame that we want to encode). To encode multiple frames in a single run, all original frames must be concatenated vertically on the same input file.

The MIP modes can be selected based on two strategies: (I) using the original samples of adjacent blocks as references for the prediction, or (II) using a smoothed version (i.e., **alternative**) of the original adjacent samples as references for the prediction. The first is faster but presents a worse coding efficiency, whereas the second poses a minor processing overhead and (generally) achieves a better coding efficiency. The selection between the two options is done by setting the **```USE_ALTERNATIVE_SAMPLES```** macro on the ```main.cpp``` file.

Running the MIP module with **original frame samples as references** uses  **```#define USE_ALTERNATIVE_SAMPLES 0```** and it takes the following form:


```bash
./main -f 2 -s 1920x1080 -o data/original_frames_0_1.csv -l MIP_decisions_log

```

Running the MIP module with **alternative samples as references** uses  **```#define USE_ALTERNATIVE_SAMPLES 1```** and it takes the following form:


```bash
./main -f 2 -s 1920x1080 -o data/original_frames_0_1.csv -l MIP_decisions_log --Filter=filterFrame_2d_float_5x5_quarterCtu --KernelIdx=2

```


where 

| Parameter | Description|
| ------ | ------ |
| f | Number of input frames to be encoded|
| s | Resolution of the frames, in the form WidthxHeight|
| o | Input file for original frame samples |
| l | Log of the MIP decisions. If left empty, no logs are created|
| Filter| Type of low-pass filter |
| KernelIdx | Index of the specific low-pass filter for the selected type |

Available filter types are listed bellow. They represent different combinations of dimension (3x3 or 5x5), representation (integer or floating-point coefficient), and implementation as separable (1D) or non-separable (2D).
| Filter type | Description|
| ------ | ------ |
| filterFrame_1d_int                     | 3x3, integer, separable |
| filterFrame_1d_float                   | 3x3, float, separable |
| filterFrame_2d_int_quarterCtu          | 3x3, integer, non-separable |
| filterFrame_2d_float_quarterCtu        | 3x3, float, non-separable |
| filterFrame_1d_int_5x5                 | 5x5, integer, separable |
| filterFrame_1d_float_5x5               | 5x5, float, separable |
| filterFrame_2d_int_5x5_quarterCtu      | 5x5, integer, non-separable |
| filterFrame_2d_float_5x5_quarterCtu    | 5x5, float, non-separable |

Kernel indices are different depending on the filter type, and can range from 0 to 4. Refer to ```convKernelLib``` in ```constants.h``` for the coefficients related to each filter type and index.

## Support for ARM devices

To run this MIP engine on ARM devices, set the macro **```#define USE_ARM 1```** on the ```main.cpp``` file.
