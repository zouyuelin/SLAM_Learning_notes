#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudacodec.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv::cuda;

__global__ void imformationfromGPU(float *x)
{
    int tidx = threadIdx.x + blockDim.x*blockIdx.x; //blockDim.x是x方向上的纬度，blockIdx.x是grid坐标，threadIdx.x是所在block的坐标
    int tidy = threadIdx.y + blockDim.y*blockIdx.y;
    x[tidx+tidy*blockDim.x*gridDim.x] = (float) tidx;
    printf("(blockIdx.x, blockIdx.y)(%d, %d) ,(tidx, tidy)(%d, %d)\n",blockIdx.x,blockIdx.y,tidx,tidy);
}

__global__ void swap_image_kernel_cvCUDA(PtrStepSz<uchar3> cu_src, PtrStepSz<uchar3> cu_dst, int h, int w)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < cu_src.cols && y < cu_src.rows)
    {
        cu_dst(y, x) = cu_src(h - y - 1, x); //cu_src(h - y - 1, x)
    }
}

__global__ void swap_image_kernel_data(uchar3 *cu_src,uchar3 *cu_dst,int h, int w )
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < w && y < h)
    {
        cu_dst[y * w + x] = cu_src[(h - y - 1)*w + x];
        if(x%100 == 0)
        printf("%d \n",cu_dst[y * w + x] );
    }
}

int main(int argc,char**argv)
{
    float *h_x,*d_x;  //h=host,d=device

    //展示二维平面
    //(x  y)分别表示水平和竖直方向上
    dim3 block(4,3);    // 4 * 3 (x y)  ---blockDim.x = 4, blockDim.y=3
    dim3 grid(3,2);     // 3 * 2       ----gridDim.x = 3 , gridDim.y = 2
    //y方向最多为6，x方向最多为12

    int nsize_ = block.x*block.y*grid.x*grid.y;

    h_x = (float *)malloc(nsize_*sizeof(float));
    cudaMalloc((void **) &d_x,nsize_*sizeof(float));

    imformationfromGPU<<<grid,block>>>(d_x);
    cudaDeviceSynchronize();//线程同步

    cudaMemcpy(h_x,d_x,nsize_*sizeof(float),cudaMemcpyDeviceToHost);

    for(int n=0;n<nsize_;n++)
    {
        printf("n,x=%d %f \n",n,h_x[n]);
    }

    cudaFree(d_x);
    free(h_x);


    //--------------------------------和OpenCV结合编程：方法1---------------------------
    GpuMat cu_src, cu_dst;
    cv::Mat src = cv::imread(argv[1]),dst;
    //开辟内存空间
    cu_src.upload(src);

    int h = src.cols;
    int w = src.rows;

    //分配新内存
    cu_dst = GpuMat(h, w, CV_8UC3, cv::Scalar(0, 0, 0));

    //把整个图像放进去
    dim3 block_(32, 32);
    dim3 grid_((w - 1) / block_.x + 1, (h +  - 1) / block_.y + 1);
    swap_image_kernel_cvCUDA<<<grid_, block_ >>> (cu_src,cu_dst,h,w);
    dst = cv::Mat(h, w, CV_8UC3, cv::Scalar(0, 0, 0));
    cu_dst.download(dst);

    cv::imshow("src",src);
    cv::imshow("cu_dst",dst);

    //--------------------------------和OpenCV结合编程：方法2---------------------------
    uchar3 *d_in;
    uchar3 *d_out;
    cudaMalloc((void**)&d_in, h*w*sizeof(uchar3));
    cudaMalloc((void**)&d_out, h*w*sizeof(uchar3));
    cudaMemcpy(d_in, src.data, h*w*sizeof(uchar3), cudaMemcpyHostToDevice);
    swap_image_kernel_data<<<grid_, block_ >>> (d_in,d_out,h,w);

    cv::Mat outputImage(h, w, CV_8UC3, cv::Scalar(0, 0, 0));//给CPU下的进行内存分配，必要步骤
    cudaMemcpy(outputImage.data, d_out, h*w*sizeof(uchar3), cudaMemcpyDeviceToHost);

    cv::imshow("outputImage",outputImage);
    cv::waitKey(0);

    return 0;
}
