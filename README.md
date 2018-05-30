# rgbTogrey
A C++ implementation of GPU accelerated Image Processing using OpenCV for conversion of any color image into its greyscale version.


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


<b><b>OPENCV_LIBPATH</b> = ```/usr/lib```
<b>OPENCV_INCLUDEPATH</b> = ```/usr/include```

# On Macs the default install locations are below

<b>#OPENCV_LIBPATH</b> = ```/usr/local/lib```
<b>#OPENCV_INCLUDEPATH</b> = ```/usr/local/include```

# or if using MacPorts

<b>#OPENCV_LIBPATH</b> = ```/opt/local/lib```
<b>#OPENCV_INCLUDEPATH</b> = ```/opt/local/include```

<b>OPENCV_LIBS</b> = ```-lopencv_core -lopencv_imgproc -lopencv_highgui```

<b>CUDA_INCLUDEPATH</b> =  ```/usr/local/cuda-5.0/include```


# On Macs the default install locations are below

<b>#CUDA_INCLUDEPATH</b> = ```/usr/local/cuda/include```
<b>#CUDA_LIBPATH</b> = ```/usr/local/cuda/lib```

<b>NVCC_OPTS</b> = ```-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64```

<b>GCC_OPTS</b> = ```-O3 -Wall -Wextra -m64```

<b>student:</b> <i>main.o student_func.o compare.o reference_calc.o Makefile</i>
	```$(NVCC) -o HW1 main.o student_func.o compare.o reference_calc.o -L $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)```

<b>main.o:</b> <i>main.cpp timer.h utils.h reference_calc.cpp compare.cpp HW1.cpp</i>
	```g++ -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH) -I $(OPENCV_INCLUDEPATH)```

<b>student_func.o:</b> <i>student_func.cu utils.h</i>
	```nvcc -c student_func.cu $(NVCC_OPTS)```

<b>compare.o:</b> <i>compare.cpp compare.h</i>
	```g++ -c compare.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)```

<b>reference_calc.o:</b> <i>reference_calc.cpp reference_calc.h</i>
	```g++ -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)```

<b>clean:</b>
	```rm -f *.o *.png hw```
