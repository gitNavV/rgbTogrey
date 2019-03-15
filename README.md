# rgbTogrey
A C++ implementation of GPU accelerated conversion of any color image into its greyscale version.


# Building on OS X

These instructions are for OS X 10.9 "Mavericks".

* Step 1. Build and install OpenCV. The best way to do this is with
Homebrew. However, you must slightly alter the Homebrew OpenCV
installation; you must build it with libstdc++ (instead of the default
libc++) so that it will properly link against the nVidia CUDA dev kit.

* Step 2. You can now create 10.9-compatible makefiles, which will allow you to
build and run your homework on your own machine:
```
mkdir build
cd build
cmake ..
make
```

<b>Instruction for make</b> :

<b>NVCC</b> = nvcc

# These are the default install
# locations on most linux distros

<b>OPENCV_LIBPATH</b> = ```/usr/lib``` </br>
<b>OPENCV_INCLUDEPATH</b> = ```/usr/include```</br>

# On Macs the default install locations are below

<b>OPENCV_LIBPATH</b> = ```/usr/local/lib```</br>
<b>OPENCV_INCLUDEPATH</b> = ```/usr/local/include```</br>

# or if using MacPorts

<b>OPENCV_LIBPATH</b> = ```/opt/local/lib```</br>
<b>OPENCV_INCLUDEPATH</b> = ```/opt/local/include```</br>

<b>OPENCV_LIBS</b> = ```-lopencv_core -lopencv_imgproc -lopencv_highgui```</br>

<b>CUDA_INCLUDEPATH</b> =  ```/usr/local/cuda-5.0/include```</br>

# On Macs the default install locations are below

<b>CUDA_INCLUDEPATH</b> = ```/usr/local/cuda/include```</br>
<b>CUDA_LIBPATH</b> = ```/usr/local/cuda/lib```</br>

<b>NVCC_OPTS</b> = ```-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64```</br>

<b>GCC_OPTS</b> = ```-O3 -Wall -Wextra -m64```</br>

<b>student:</b> <i>main.o student_func.o compare.o reference_calc.o Makefile</i></br>
	```$(NVCC) -o HW1 main.o student_func.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)```</br>

<b>main.o:</b> <i>main.cpp timer.h utils.h reference_calc.cpp compare.cpp HW1.cpp</i></br>
	```g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)```</br>

<b>student_func.o:</b> <i>student_func.cu utils.h</i></br>
	```nvcc -c student_func.cu $(NVCC_OPTS)```</br>

<b>compare.o:</b> <i>compare.cpp compare.h</i></br>
	```g++ -c compare.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)```</br></br>

<b>reference_calc.o:</b> <i>reference_calc.cpp reference_calc.h</i></br>
	```g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)```</br>

<b>clean:</b></br>
	```rm -f *.o *.png hw```
