#include <iostream>
#include <string>
#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>>
#include "compare.h"
#include "reference_calc.h"
#include "utils.h"
#include "timer.h"

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4 *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

void preProcess(uchar4 **inputImage, unsigned char **greyImage, uchar4 **d_rgbaImage, unsigned char **d_greyImage, const std::string &filename)
{
  
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) 
  {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  imageGrey.create(image.rows, image.cols, CV_8UC1);

  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) 
  {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char)));

  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

__global__ void rgba_to_greyscale(const uchar4* const rgbaImage, unsigned char* const greyImage, int numRows, int numCols)
{
  int y = threadIdx.y+ blockIdx.y* blockDim.y;
  int x = threadIdx.x+ blockIdx.x* blockDim.x;

  if (y < numCols && x < numRows) {
    int index = numRows*y + x;
  uchar4 color = rgbaImage[index];
  unsigned char grey = (unsigned char)(0.299f*color.x+ 0.587f*color.y + 0.114f*color.z);
  greyImage[index] = grey;
  }
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage, unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  int blockWidth = 32;

  const dim3 blockSize(blockWidth, blockWidth, 1);

  int blocksX = numRows/blockWidth+1;
  int blocksY = numCols/blockWidth+1;

  const dim3 gridSize( blocksX, blocksY, 1);

  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
}

void postProcess(const std::string& output_file, unsigned char* data_ptr) {
  cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);

  cv::imwrite(output_file.c_str(), output);
}

void cleanup()
{
  cudaFree(d_rgbaImage__);
  cudaFree(d_greyImage__);
}

void compareImages(std::string reference_filename, std::string test_filename, bool useEpsCheck, double perPixelError, double globalError)
{
  cv::Mat reference = cv::imread(reference_filename, -1);
  cv::Mat test = cv::imread(test_filename, -1);

  cv::Mat diff = abs(reference - test);

  cv::Mat diffSingleChannel = diff.reshape(1, 0); 

  double minVal, maxVal;

  cv::minMaxLoc(diffSingleChannel, &minVal, &maxVal, NULL, NULL); 


  diffSingleChannel = (diffSingleChannel - minVal) * (255. / (maxVal - minVal));

  diff = diffSingleChannel.reshape(reference.channels(), 0);

  cv::imwrite("HW1_differenceImage.png", diff);
  unsigned char *referencePtr = reference.ptr<unsigned char>(0);
  unsigned char *testPtr = test.ptr<unsigned char>(0);

  if (useEpsCheck) 
  {
    checkResultsEps(referencePtr, testPtr, reference.rows * reference.cols * reference.channels(), perPixelError, globalError);
  }

  else
  {
    checkResultsExact(referencePtr, testPtr, reference.rows * reference.cols * reference.channels());
  }

  std::cout << "PASS" << std::endl;
  return;
}


void generateReferenceImage(std::string input_filename, std::string output_filename)
{
  cv::Mat reference = cv::imread(input_filename, CV_LOAD_IMAGE_GRAYSCALE);

  cv::imwrite(output_filename, reference);

}

int main(int argc, char **argv) 
{
  uchar4 *h_rgbaImage, *d_rgbaImage;
  unsigned char *h_greyImage, *d_greyImage;

  std::string input_file;
  std::string output_file;
  std::string reference_file;
  double perPixelError = 0.0;
  double globalError = 0.0;
  bool useEpsCheck = false;
  switch (argc)
  {
	case 2:
	  input_file = std::string(argv[1]);
	  output_file = "HW1_output.png";
	  reference_file = "HW1_reference.png";
	  break;
	case 3:
	  input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
	  reference_file = "HW1_reference.png";
	  break;
	case 4:
	  input_file  = std::string(argv[1]);
    output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  break;
	case 6:
	  useEpsCheck=true;
	  input_file  = std::string(argv[1]);
	  output_file = std::string(argv[2]);
	  reference_file = std::string(argv[3]);
	  perPixelError = atof(argv[4]);
    globalError   = atof(argv[5]);
	  break;
	default:
      std::cerr << "Usage: ./HW1 input_file [output_filename] [reference_filename] [perPixelError] [globalError]" << std::endl;
      exit(1);
  }

  preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

  GpuTimer timer;
  timer.Start();

  your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage, numRows(), numCols());

  timer.Stop();

  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  int err = printf("Your code ran in: %f msecs.\n", timer.Elapsed());

  if (err < 0) {
    std::cerr << "Couldn't print timing information! STDOUT Closed!" << std::endl;
    exit(1);
  }

  size_t numPixels = numRows() * numCols();
  
  checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

  postProcess(output_file, h_greyImage);

  referenceCalculation(h_rgbaImage, h_greyImage, numRows(), numCols());

  postProcess(reference_file, h_greyImage);

  compareImages(reference_file, output_file, useEpsCheck, perPixelError, globalError);

  cleanup();

  return 0;
}
