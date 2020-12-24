#include <iostream>
#include <time.h>
#include <stdio.h>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

/*
    Compile and run instructions: Do not use nvcc. Use the two commands below to compile.
    cmake .
    make
    
    To run the program =>   ./cudaProj
*/

#define THREADS_PER_BLOCK 512 
#define N (4*4)

#define TILE_DIM 32;
#define BLOCK_ROWS 8;
#define NUM_REPS 100;

__global__ void transpose(int *a, int *b){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
  	int y = blockIdx.y * blockDim.x + threadIdx.y;

  	int width = gridDim.x * TILE_DIM;

  	for (int j = 0; j < TILE_DIM; j+= BLOCK_ROWS)
    		a[(y+j)*width + x] = b[(y+j)*width + x];
}

__global__ void transposeImage(uchar3 * const d_in, unsigned char * const d_out, 
				uint imgheight, uint imgwidth){
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    	const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

        for (int j = 0; j < blockDim.x; j+= imgwidth){
                d_out[idx*imgwidth + (idy+j)] = d_in[(idy+j)*imgwidth + idx];
        }

}

int main(int argc, char **argv){
	const int nx = 1024;
  	const int ny = 1024;
  	const int mem_size = nx*ny*sizeof(float);

  	dim3 dimGrid(nx/TILE_DIM, ny/TILE_DIM, 1);
  	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	float *d_adata = (float*)malloc(mem_size);
	float *d_bdata = (float*)malloc(mem_size);	

	transpose<<<dimGrid, dimBlock>>>(d_tdata, d_idata);

	for (int i = 0; i < NUM_REPS; i++)
     		transposeNaive<<<dimGrid, dimBlock>>>(d_tdata, d_idata);	
	
	free(d_adata);
  	free(d_bdata);

    Mat srcImage = imread("./e1.jpg");
    const uint imgheight = srcImage.rows;
    const uint imgwidth = srcImage.cols;

	Mat inputImage(imgheight, imgwidth, CV_8UC3);
	Mat outputImage(imgwidth, imgheight , CV_8UC3);

    uchar3 *d_in;
    unsigned char *d_out;

    cudaMalloc((void**)&d_in, imgheight*imgwidth*sizeof(uchar3));
    cudaMalloc((void**)&d_out, imgheight*imgwidth*sizeof(unsigned char));

    cudaMemcpy(d_in, srcImage.data, imgheight*imgwidth*sizeof(uchar3), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgwidth + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (imgheight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transposeImage<< <blocksPerGrid, threadsPerBlock>> >(d_in, d_out, imgheight, imgwidth);

    cudaMemcpy(outputImage.data, d_out, imgheight*imgwidth*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
    cudaFree(d_out);

	imwrite("transposeImage.jpg",outputImage);
	
    return 0;

}
