#all: kernel main native main_5
all: native 

native: convnet.h  convolutional_layer.h  fullyconnected_layer.h  layer.h  maxpooling_layer.h  mnist_parser.h  output_layer.h  settings.h  util.h main.cpp cl_wrapper.hpp  timer.hpp filemgt.hpp
	g++ -m32 -o main_native main.cpp -I/opt/AMDAPP/include/ -L/opt/AMDAPP/lib/x86 -lOpenCL -pthread -ldl -std=c++11

main: convnet.h  convolutional_layer.h  fullyconnected_layer.h  layer.h  maxpooling_layer.h  mnist_parser.h  output_layer.h  settings.h  util.h main.cpp cl_wrapper.hpp  timer.hpp filemgt.hpp
	g++ -m32 -o main main.cpp -I/home/xujiao/multi2sim-4.2/runtime/include -L/home/xujiao/multi2sim-4.2/lib/.libs -lm2s-opencl -static -pthread -ldl -std=c++11

main_5: convnet.h  convolutional_layer.h  fullyconnected_layer.h  layer.h  maxpooling_layer.h  mnist_parser.h  output_layer.h  settings.h  util.h main.cpp cl_wrapper.hpp  timer.hpp filemgt.hpp
	g++ -m32 -o main_5 main.cpp -I/home/shepherd/m2s5/multi2sim-5.0/runtime/include -L/home/shepherd/m2s5/multi2sim-5.0/lib/.libs -lm2s-opencl -static -pthread -ldl -std=c++11

kernel: kernels.ocl
	m2c --amd --amd-device Tahiti kernels.ocl # Tahiti: Southern Island

clean:
	rm -rf main kernels.bin main_native main_5
