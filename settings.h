// 1. Requisite files
#define DATA_PATH "mnist"
#define KERNEL_PATH "kernels.ocl"
 
// 2. Build target/env control
#define GPU
//#define PROFILING
//#define BATCH_MORE
//#define CHECK_RESULT
//#define THREAD_TASKS 3

// 3. Workload control
#define MAX_ITER 1 // maximum training iterations
#define M 10 // training sample counts in each iteration
#define END_CONDITION 1e-3 // Not used due to exception
#define DEF_TEST_SAMPLE_COUNT 1000

// 4. NN file control
#define NN_DATA_FILENAME "nn_learned"
#define NN_DATA_FILENAME_EXT "cnn"
//#define SAVE_INTER_NN 1000
//#define SAVE_FINAL_MN

// 5. training & testing control
#define SKIP_TRAINING
//#define PARTIAL_TEST_IMG 100
