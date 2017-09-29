#ifndef __JC_UTIL_H__
#define __JC_UTIL_H__

#include <exception>
#include <fstream>
#include <sstream>
#include <stdexcept>

// Yeseong
// C++ implementation (not supported)
//#define __CL_ENABLE_EXCEPTIONS
//#include "CL/cl.hpp"

#include <stdio.h>
#include <CL/cl.h>

inline void check_body(cl_int status, int line, const char* filename) {
	if (status == CL_SUCCESS)
		return;

	printf("%s:%d\tError %d\n", filename, line, status);
	exit(0);
}

#define CHECK(STATUS) check_body(STATUS, __LINE__, __FILE__)

namespace cl_wrapper { 
// Yeseong: Moved from Khronos Group's c++ implementation
//! \brief Class interface for specifying NDRange values.
class NDRange
{
	private:
		size_t sizes_[3];
		cl_uint dimensions_;

	public:
		//! \brief Default constructor - resulting range has zero dimensions.
		NDRange()
			: dimensions_(0)
		{ }   

		//! \brief Constructs one-dimensional range.
		NDRange(::size_t size0)
			: dimensions_(1)
		{
			sizes_[0] = size0;
		}

		//! \brief Constructs two-dimensional range.
		NDRange(::size_t size0, ::size_t size1)
			: dimensions_(2)
		{
			sizes_[0] = size0;
			sizes_[1] = size1;
		}

		//! \brief Constructs three-dimensional range.
		NDRange(::size_t size0, ::size_t size1, ::size_t size2)
			: dimensions_(3)
		{
			sizes_[0] = size0;
			sizes_[1] = size1;
			sizes_[2] = size2;
		}

		/*! \brief Conversion operator to const ::size_t *.
		 *  
		 *  \returns a pointer to the size of the first dimension.
		 */
		operator const ::size_t*() const { 
			return (const ::size_t*) sizes_; 
		}

		//! \brief Queries the number of dimensions in the range.
		size_t dimensions() const { return dimensions_; }
};

//! \brief A zero-dimensional range.
static const NDRange NullRange;

// Yeseong: Wrapper function for buffer
cl_mem createBuffer(const cl_context& context, cl_mem_flags flags, size_t size) {
	cl_int error;
	return clCreateBuffer(context, flags, size, NULL, NULL);
}

template<typename T>
void setArg(cl_kernel& kernel, cl_uint arg_index, T& arg_value)
{
	cl_int status = clSetKernelArg(kernel, arg_index, sizeof(T), (const void*)&arg_value);
	CHECK(status);
}

}

namespace jc {

std::string fileToString(const std::string& file_name) {
    std::string file_text;

    std::ifstream file_stream(file_name.c_str());
    if (!file_stream) {
        std::ostringstream oss;
        oss << "There is no file called " << file_name;
        throw std::runtime_error(oss.str());
    }

    file_text.assign(std::istreambuf_iterator<char>(file_stream), std::istreambuf_iterator<char>());

    return file_text;
}

cl_program buildProgram(const std::string& file_name, const cl_context& context, const cl_device_id* devices)
{
    std::string source_code = jc::fileToString(file_name);
	size_t source_size = source_code.size();
	const char* source_str = source_code.c_str();

	cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_size, NULL);
	cl_int status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
	CHECK(status);
	
	// TODO: Destroy program

    return program;
}

cl_ulong runAndTimeKernel(const cl_kernel& kernel, const cl_command_queue& queue, const cl_wrapper::NDRange global, const cl_wrapper::NDRange& local=cl_wrapper::NullRange)
{
    cl_ulong t1, t2;
    cl_event evt;
	
	cl_int status = clEnqueueNDRangeKernel(
			queue, kernel, (cl_uint) global.dimensions(),
			NULL, // offset is not sed
			(const ::size_t*) global,
			local.dimensions() != 0 ? (const ::size_t*) local : NULL,
			0, NULL, &evt);
	CHECK(status);
	
	clWaitForEvents(1, &evt);

	clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &t1, NULL);
	clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &t2, NULL);

    return t2 - t1;
}

const char *readable_status(cl_int status)
{
    switch (status) {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
#ifndef CL_VERSION_1_0
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
            return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: 
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
#endif
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        case CL_INVALID_GLOBAL_WORK_SIZE:
            return "CL_INVALID_GLOBAL_WORK_SIZE";
#ifndef CL_VERSION_1_0
        case CL_INVALID_PROPERTY:
            return "CL_INVALID_PROPERTY";
#endif
        default:
            return "CL_UNKNOWN_CODE";
    }
}

unsigned int closestMultiple(unsigned int size, unsigned int divisor)
{
    unsigned int remainder = size % divisor;
    return remainder == 0 ? size : size - remainder + divisor;
}

template <class T>
void showMatrix(T *matrix, unsigned int width, unsigned int height)
{
    for (unsigned int row = 0; row < height; ++row) {
        for (unsigned int col = 0; col < width; ++col) {
            std::cout << matrix[width*row + col] << " ";
        }
        std::cout << std::endl;
    }
    return;
}

}

#endif
