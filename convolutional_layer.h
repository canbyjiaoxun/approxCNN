#pragma once
#include "util.h"
#include "layer.h"

#include "cl_wrapper.hpp"

namespace convnet{
	class ConvolutionalLayer :public Layer
	{
	public:
		ConvolutionalLayer(size_t in_width, size_t in_height, size_t in_depth, 
			size_t kernel_size, size_t out_depth) :
			Layer(in_width, in_height, in_depth, in_width - kernel_size + 1, in_height - kernel_size + 1, out_depth, 0.3, 0.01),
			kernel_size_(kernel_size)
		{
			W_.resize(kernel_size * kernel_size * in_depth_ * out_depth_);
			deltaW_.resize(kernel_size * kernel_size * in_depth_ * out_depth_);
			b_.resize(out_depth * out_width_* out_height_);
			output_.resize(out_depth * out_width_ * out_height_);
			this->init_weight();
            this->init_opencl();
		}

		void init_weight(){
			uniform_rand(W_.begin(), W_.end(), -1, 1);
			uniform_rand(b_.begin(), b_.end(), -1, 1);
		}
       
        cl_context context;
        cl_command_queue queue;
        cl_program program;

        void init_opencl(){
#ifndef GPU
			return;
#endif
            // OpenCL initialization  
            // std::vector<cl::Platform> platforms;
            // std::vector<cl::Device> devices;
            // cl::Platform::get(&platforms);
            // platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

			cl_uint numPlatforms;
			cl_platform_id platform = NULL;

			cl_int status = clGetPlatformIDs(0, NULL, &numPlatforms);
			CHECK(status);

			cl_platform_id* platforms = (cl_platform_id*)malloc(numPlatforms* sizeof(cl_platform_id));
			status = clGetPlatformIDs(numPlatforms, platforms, NULL);
			CHECK(status);

			platform = platforms[0]; // Pick the first available platform
			free(platforms);

			// Prepare a device
			cl_uint	numDevices = 0;
			cl_device_id* devices = NULL;

			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);	
			CHECK(status);
			if (numDevices == 0) {	//no GPU available.
				printf("No GPU device available.\n");
				CHECK(-1);
			}

			devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
			status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);
			CHECK(status);


            // context = cl::Context(devices);
            // queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);
            // program = jc::buildProgram(KERNEL_PATH, context, devices);

			// Create context and command quque
			context = clCreateContext(NULL, 1, devices, NULL, NULL, &status);
			CHECK(status);
			queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &status);
			//queue = clCreateCommandQueue(context, devices[0], 0, &status);
			CHECK(status);
            program = jc::buildProgram(KERNEL_PATH, context, devices);

			// TODO FREEING to malloc objects
			//status = clReleaseProgram(program);             //Release the program object.
			//status = clReleaseCommandQueue(commandQueue);   //Release  Command queue.
			//status = clReleaseContext(context);             //Release context.
        }

		void forward(){
            forward_cpu();
		}

        void forward_cpu(){  
            std::fill(output_.begin(), output_.end(), 0);
            for (size_t out = 0; out < out_depth_; out++){  /* for each output feature map */
            	for (size_t in = 0; in < in_depth_; in++){  /* for each input feature map */
            		for (size_t h_ = 0; h_ < out_height_; h_++){
            			for (size_t w_ = 0; w_ < out_width_; w_++){
            				output_[getOutIndex(out, h_, w_)] +=
            					conv(getInforKernel(in, h_, w_), getW_(in, out));
            			}
            		}
            	}

				/* use activate function to get output */
				for (size_t h_ = 0; h_ < out_height_; h_++){
            		for (size_t w_ = 0; w_ < out_width_; w_++){
						//std::cout << sigmod(output_[getOutIndex(out, h_, w_)] + /*eh?*/ b_[getb_(out, h_, w_)])<< "\t" << output_[getOutIndex(out, h_, w_)] + /*eh?*/ b_[getb_(out, h_, w_)]  << std::endl;

                        output_[getOutIndex(out, h_, w_)] =
                            sigmod(output_[getOutIndex(out, h_, w_)] + /*eh?*/ b_[getb_(out, h_, w_)]);
            		}
            	}
            }

			//for (auto it = output_.begin(); it != output_.end(); ++it) {
			//	std::cout << *it << std::endl;
			//}
        }

        void forward_gpu(){
            try {
            // Allocate memory on the device
            // cl::Buffer input_buf(context, CL_MEM_READ_ONLY, in_width_*in_height_*in_depth_*sizeof(float_tt));
            // cl::Buffer weight_buf(context, CL_MEM_READ_ONLY, kernel_size_*kernel_size_*in_depth_*out_depth_*sizeof(float_tt));
            // cl::Buffer b_buf(context, CL_MEM_READ_ONLY, out_depth_ * out_width_* out_height_*sizeof(float_tt));
            // cl::Buffer output_buf(context, CL_MEM_WRITE_ONLY, out_width_*out_height_*out_depth_*sizeof(float_tt));
			cl_mem input_buf = cl_wrapper::createBuffer(context, CL_MEM_READ_ONLY, in_width_*in_height_*in_depth_*sizeof(float_tt));
			cl_mem weight_buf= cl_wrapper::createBuffer(context, CL_MEM_READ_ONLY, kernel_size_*kernel_size_*in_depth_*out_depth_*sizeof(float_tt));
			cl_mem b_buf = cl_wrapper::createBuffer(context, CL_MEM_READ_ONLY, out_depth_ * out_width_* out_height_*sizeof(float_tt));
			cl_mem output_buf = cl_wrapper::createBuffer(context, CL_MEM_WRITE_ONLY, out_width_*out_height_*out_depth_*sizeof(float_tt));

            std::string kernel_name = "forward_parallel";
            cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), NULL);
            cl_wrapper::setArg<cl_mem>(kernel, 0, input_buf);
            cl_wrapper::setArg<cl_mem>(kernel, 1, weight_buf);
            cl_wrapper::setArg<cl_mem>(kernel, 2, b_buf);
            cl_wrapper::setArg<cl_mem>(kernel, 3, output_buf);
            cl_wrapper::setArg<size_t>(kernel, 4, in_width_);
            cl_wrapper::setArg<size_t>(kernel, 5, in_height_);
            cl_wrapper::setArg<size_t>(kernel, 6, in_depth_);
            cl_wrapper::setArg<size_t>(kernel, 7, out_width_);
            cl_wrapper::setArg<size_t>(kernel, 8, out_height_);
            cl_wrapper::setArg<size_t>(kernel, 9, out_depth_);
            cl_wrapper::setArg<size_t>(kernel, 10, kernel_size_);

            // transfer source data from the host to the device
			cl_event wevt1, wevt2, wevt3, revt1;
            cl_int status = clEnqueueWriteBuffer(queue, input_buf, CL_TRUE, 0, in_width_*in_height_*in_depth_*sizeof(float_tt), &input_[0], 0, NULL, &wevt1);
			CHECK(status);
			clWaitForEvents(1, &wevt1);
			status = clEnqueueWriteBuffer(queue, weight_buf, CL_TRUE, 0, kernel_size_*kernel_size_*in_depth_*out_depth_*sizeof(float_tt), &W_[0], 0, NULL, &wevt2);
			CHECK(status);
			clWaitForEvents(1, &wevt2);
			status = clEnqueueWriteBuffer(queue, b_buf, CL_TRUE, 0, out_depth_ * out_width_* out_height_*sizeof(float_tt), &b_[0], 0, NULL, &wevt3);
			CHECK(status);
			clWaitForEvents(1, &wevt3);
            
            // execute the code on the device
            int grpWidth = 1;
            cl_wrapper::NDRange global(jc::closestMultiple(out_depth_*out_width_, grpWidth), 
                               jc::closestMultiple(out_height_, grpWidth));
            cl_wrapper::NDRange local(grpWidth, grpWidth);

			//std::cout << out_depth_*out_width_ << std::endl;
			//std::cout << out_height_ << std::endl;
			//std::cout << ((const :: size_t*)global)[0] << std::endl;
			//std::cout << ((const :: size_t*)global)[1] << std::endl;
			//std::cout << ((const :: size_t*)local)[0] << std::endl;
			//std::cout << ((const :: size_t*)local)[1] << std::endl;
			
            cl_ulong t = jc::runAndTimeKernel(kernel, queue, global, local);

            // transfer destination data from the device to the host
            status = clEnqueueReadBuffer(queue, output_buf, CL_TRUE, 0, out_width_*out_height_*out_depth_*sizeof(float_tt), &output_[0], 0, NULL, &revt1);
			CHECK(status);
			clWaitForEvents(1, &revt1);

			// Release resources
			clFinish(queue);
			status = clReleaseKernel(kernel); 
			CHECK(status);
			status = clReleaseMemObject(input_buf);
			CHECK(status);
			status = clReleaseMemObject(weight_buf);
			CHECK(status);
			status = clReleaseMemObject(b_buf);
			CHECK(status);
			status = clReleaseMemObject(output_buf);
			CHECK(status);

			//int i = 0;
			//for (auto it = output_.begin(); it != output_.end(); ++it) {
			//	std::cout << i << "\t" << (int)(*it * 10000) << std::endl;
			//	++i;
			//}
			//std::cout << "LAYER" << std::endl;
			//std::cout << (int)(sigmod(b_[86])* 1000) << std::endl;
		}
        // catch (cl::Error& e) {
        //     std::cerr << e.what() << ": " << jc::readable_status(e.err());
        //     //return 3;
        // }
        catch (std::exception& e) {
            std::cerr << e.what() << std::endl;
            //return 2;
        }
        catch (...) {
            std::cerr << "Unexpected error. Aborting!\n" << std::endl;
            //return 1;
        }

        }

        void forward_batch(int batch_size){

            try {
                // Allocate memory on the device
                cl_mem input_batch_buf = cl_wrapper::createBuffer(context, CL_MEM_READ_ONLY, batch_size*in_width_*in_height_*in_depth_*sizeof(float_tt));
                cl_mem weight_buf = cl_wrapper::createBuffer(context, CL_MEM_READ_ONLY, kernel_size_*kernel_size_*in_depth_*out_depth_*sizeof(float_tt));
                cl_mem b_buf = cl_wrapper::createBuffer(context, CL_MEM_READ_ONLY, out_depth_ * out_width_* out_height_*sizeof(float_tt));
                cl_mem output_batch_buf = cl_wrapper::createBuffer(context, CL_MEM_WRITE_ONLY, batch_size*out_width_*out_height_*out_depth_*sizeof(float_tt));

#ifdef BATCH_MORE
                std::string kernel_name = "forward_batch_more";
#else
                std::string kernel_name = "forward_batch";
                //std::string kernel_name = "forward_parallel";
#endif
				std::cout << kernel_name << std::endl;
                cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), NULL);
                cl_wrapper::setArg<cl_mem>(kernel, 0, input_batch_buf);
                cl_wrapper::setArg<cl_mem>(kernel, 1, weight_buf);
                cl_wrapper::setArg<cl_mem>(kernel, 2, b_buf);
                cl_wrapper::setArg<cl_mem>(kernel, 3, output_batch_buf);
                cl_wrapper::setArg<size_t>(kernel, 4, in_width_);
                cl_wrapper::setArg<size_t>(kernel, 5, in_height_);
                cl_wrapper::setArg<size_t>(kernel, 6, in_depth_);
                cl_wrapper::setArg<size_t>(kernel, 7, out_width_);
                cl_wrapper::setArg<size_t>(kernel, 8, out_height_);
                cl_wrapper::setArg<size_t>(kernel, 9, out_depth_);
                cl_wrapper::setArg<size_t>(kernel, 10, kernel_size_);
                cl_wrapper::setArg<int>(kernel, 11, batch_size);

                // transfer source data from the host to the device
                clEnqueueWriteBuffer(queue, input_batch_buf, CL_TRUE, 0, batch_size*in_width_*in_height_*in_depth_*sizeof(float_tt), &input_batch_[0], 0, NULL, NULL);
                clEnqueueWriteBuffer(queue, weight_buf, CL_TRUE, 0, kernel_size_*kernel_size_*in_depth_*out_depth_*sizeof(float_tt), &W_[0], 0, NULL, NULL);
                clEnqueueWriteBuffer(queue, b_buf, CL_TRUE, 0, out_depth_ * out_width_* out_height_*sizeof(float_tt), &b_[0], 0, NULL, NULL);

                // execute the code on the device
                int grpWidth = 20;

                int global_width = jc::closestMultiple(out_depth_*out_width_, grpWidth);
#ifdef BATCH_MORE
                int global_height = jc::closestMultiple((batch_size+THREAD_TASKS-1)/THREAD_TASKS*out_height_, grpWidth);
#else
                int global_height = jc::closestMultiple(batch_size*out_height_, grpWidth);
#endif
                cl_wrapper::NDRange global(global_width, global_height);
                cl_wrapper::NDRange local(grpWidth, grpWidth);

#ifndef PROFILING
                cl_ulong t = jc::runAndTimeKernel(kernel, queue, global, local);
				std::cout << "End1" << std::endl;
#else
                int iteration = 100;                
                int input_data_size = (batch_size*in_width_*in_height_*in_depth_
                    + kernel_size_*kernel_size_*in_depth_*out_depth_
                    + batch_size*out_depth_ * out_width_* out_height_)*sizeof(float_tt);
                int output_data_size = batch_size*out_width_*out_height_*out_depth_*sizeof(float_tt);
#ifdef BATCH_MORE
                printf(" **** In ConvolutionalLayer::forward_batch_more ****\n");
                int memory_access_per_thread = (in_depth_*kernel_size_*kernel_size_*(1+THREAD_TASKS) + THREAD_TASKS)*sizeof(float);
                int operations = in_depth_*kernel_size_*kernel_size_*9
                                    + in_depth_*THREAD_TASKS*kernel_size_*kernel_size_*15 + THREAD_TASKS*20;
                printf("    Batch size: %d, Tasks of each thread: %d\n    INPUT depth: %d, height: %d, width: %d\n    OUTPUT depth: %d, height: %d, width: %d\n",
                    batch_size, THREAD_TASKS, in_depth_, in_height_, in_width_, out_depth_, out_height_, out_width_);
#else
                printf(" **** In ConvolutionalLayer::forward_batch ****\n");
                int memory_access_per_thread = (in_depth_ * 2 * kernel_size_*kernel_size_ + 1 + 1)*sizeof(float);
                int operations = 22 + 26 * in_depth_*kernel_size_*kernel_size_;
                printf("    Batch size: %d\n    INPUT depth: %d, height: %d, width: %d\n    OUTPUT depth: %d, height: %d, width: %d\n",
                    batch_size, in_depth_, in_height_, in_width_, out_depth_, out_height_, out_width_);
#endif
                
                
                printf("    ==Running with>>> %d <<<Iterations==\n", iteration);

                cl_ulong t = 0; // time in nanosecond, 1e-9 second
                for (int i = 0; i < iteration; i++)
                    t += jc::runAndTimeKernel(kernel, queue, global, local);

                const float each_lasts = float(t) / iteration; // nano seconds
                std::cout << "    Time consumed for each iteration: " << each_lasts / 1e6 << " ms" << std::endl;
                std::cout << "    Time consumed for each batch: " << each_lasts / batch_size / 1e6 << " ms" << std::endl;
                float cpI = float(operations) / memory_access_per_thread;
                float peak_bandwidth = 25.6; // Memory Bandwidth: 25.6 GB/s
#ifdef BATCH_MORE
                float throughPut = memory_access_per_thread * batch_size*out_depth_*out_width_*out_height_ / THREAD_TASKS / each_lasts ; // GB/s
                long long int all_ops = operations*out_depth_*out_width_*out_height_*(batch_size + THREAD_TASKS -1) / THREAD_TASKS;
#else
                float throughPut = memory_access_per_thread * batch_size*out_depth_*out_width_*out_height_ / each_lasts; // GB/s
                long long int all_ops = operations*out_depth_*out_width_*out_height_*batch_size;
#endif
                printf("    Input Buffer size: %.2g MB, Output Buffer size: %.2g MB\n", input_data_size / 1e6, output_data_size / 1e6);
                printf("    CI: %.2g, ThoughPut: %.3g GB/s, Ops/Time= %.3g GFLOPS, CI*Bandwidth= %.3g GFLOPS\n",
                       cpI, throughPut, all_ops/each_lasts, cpI*peak_bandwidth);
#endif
                output_batch_.resize(batch_size*out_depth_ * out_width_ * out_height_);
                // transfer destination data from the device to the host
				std::cout << "End2" << std::endl;
                clEnqueueReadBuffer(queue, output_batch_buf, CL_TRUE, 0, batch_size*out_width_*out_height_*out_depth_*sizeof(float_tt), &output_batch_[0], 0, NULL, NULL);
				std::cout << "End3" << std::endl;
            }
            // catch (cl::Error& e) {
            //     std::cerr << e.what() << ": " << jc::readable_status(e.err());
            //     //return 3;
            // }
            catch (std::exception& e) {
                std::cerr << e.what() << std::endl;
                //return 2;
            }
            catch (...) {
                std::cerr << "Unexpected error. Aborting!\n" << std::endl;
                //return 1;
            }

        }


		void back_prop(){
			g_.clear();
			g_.resize(in_width_ * in_height_ * in_depth_);
			/*update err terms of this layer.*/
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					for (size_t w_ = 0; w_ < out_width_; w_++){
						for (size_t h_ = 0; h_ < out_height_; h_++){
							for (size_t y_ = 0; y_ < kernel_size_; y_++){
								for (size_t x_ = 0; x_ < kernel_size_; x_++){
									auto ff = in * in_width_ * in_height_ + (h_ + y_) *
										in_width_ + (x_ + w_);
									g_[ff] += /*next layer err terms*/
										this->next->g_[out * out_width_ *
										out_height_ + h_ * out_width_ + w_] * 
										/*weight*/
										W_[in * out_depth_ * kernel_size_ * kernel_size_ +
                                           out * kernel_size_ * kernel_size_ +
                                           kernel_size_ * (kernel_size_ - y_ - 1) +
                                           (kernel_size_ - 1 - x_)] *
										/*df of input*/
										df_sigmod(input_[ff]);
								}
							}
						}
					}
				}
			}

			/*update weight*/
			for (size_t out = 0; out < out_depth_; out++){
				for (size_t in = 0; in < in_depth_; in++){
					for (size_t h_ = 0; h_ < out_height_; h_++){
						for (size_t w_ = 0; w_ < out_height_; w_++){
							auto tt = getb_(out, h_, w_);
							for (size_t y_ = 0; y_ < kernel_size_; y_++){
								for (size_t x_ = 0; x_ < kernel_size_; x_++){
									/*find update pixel*/
									auto target = in * out_depth_ * kernel_size_ * kernel_size_ +
										out * kernel_size_ * kernel_size_ +
										kernel_size_ * (kernel_size_ - y_ - 1) +
										(kernel_size_ - 1 - x_);
									/*cal delta*/
									auto delta =
                                        /*learning rate*/
										alpha_ *
										/*input*/
										input_[in * in_width_ * in_height_ + (h_ + y_) *
                                               in_width_ + (x_ + w_)] *
										/*next layer err terms*/
										this->next->g_[tt]
										/*weight momentum*/
										+ lambda_ * deltaW_[target];
										
                                    W_[target] += delta;
                                    /*update momentum*/
                                    deltaW_[target] = delta;
								}
							}
							b_[tt] += alpha_ * this->next->g_[tt];
						}
					}
				}
			}
		}

	private:
		inline size_t getOutIndex(size_t out, size_t h_, size_t w_){
			return out * out_height_ * out_width_ + h_ * out_width_ + w_;
		}

		inline vec_t getInforKernel(size_t in, size_t h_, size_t w_){
			vec_t r;
			for (size_t y = 0; y < kernel_size_; y++){
				for (size_t x = 0; x < kernel_size_; x++){
					r.push_back(input_[in * (in_width_ * in_height_) + (h_ + y) * in_width_ + x + w_]);
				}
			}
			return r;
		}

		inline vec_t getW_(size_t in, size_t out){
			vec_t r;
			for (size_t i = 0; i < kernel_size_ * kernel_size_; i++)
				r.push_back(W_[in * out_depth_ * kernel_size_ * kernel_size_ 
				+ out * kernel_size_ * kernel_size_ + i]);
			return r;
		}

		inline int getb_(size_t out, size_t h_, size_t w_){
			return out * out_width_ * out_height_ + h_ * out_height_ + w_;
		}

		/*
		2-dimension convoluton:

			1 2 3                    1 -1 0
			3 4 2  conv with kernel  -1 0 1  
			2 1 3                    1  1 0

			---->
			1*0 + 2*1 + 3*1 + 3*1 + 4*0 + 2*-1 + 2*0 + 1*-1 + 3*1
			return the sum.

		see also:
		*/
		float_tt conv(vec_t a, vec_t b){
			assert(a.size() == b.size());
			float_tt sum = 0, size = a.size();
			for (size_t i = 0; i < size; i++){
				sum += a[i] * b[size - i - 1];
			}
			return sum;
		}

		size_t kernel_size_;
	};
}// namespace convnet
