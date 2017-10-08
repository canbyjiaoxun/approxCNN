#include "convnet.h"
#include <unistd.h>
#include <CL/cl.h>

using namespace std;
using namespace convnet;

int main(int argc, char* argv[]){
	std::string strLoadFilename = filemgt::get_default_filename();
	if (argc >= 2) {
		strLoadFilename = argv[1];
	}

	int test_sample_count = DEF_TEST_SAMPLE_COUNT;
	if (argc == 3) {
		test_sample_count = atoi(argv[2]);
	}

	Mnist_Parser m(DATA_PATH);
	
	printf("Data path: %s\n", DATA_PATH);
//	printf("%lf\n", c);
//	printf("%d %d\n", clGetDeviceInfo(), clGetPlatformInfo());
	// Training ===================================
#ifndef SKIP_TRAINING
	m.load_training();
	std::uint32_t num_train = m.train_sample.size();
	std::cout << "Image mapping: " << num_train << std::endl;

	vec2d_t x;
	vec_t y;
	for (size_t i = 0; i < num_train; i++) {
		x.push_back(m.train_sample[i]->image);
		y.push_back(m.train_sample[i]->label);
	}
#endif // SKIP_TRAINING

	std::cout << "Conv Create" << std::endl;
	ConvNet n;

	n.add_layer(new ConvolutionalLayer(32, 32, 1, 5, 6));
	n.add_layer(new MaxpoolingLayer(28, 28, 6));
	n.add_layer(new ConvolutionalLayer(14, 14, 6, 5, 16));
	n.add_layer(new MaxpoolingLayer(10, 10, 16));
	n.add_layer(new ConvolutionalLayer(5, 5, 16, 5, 100));
	n.add_layer(new FullyConnectedLayer(100, 10));
	n.add_layer(new OutputLayer(10)); // Yeseong: moved from convnet.h

	std::cout << "Conv Train" << std::endl;

	// Load layers
	int prelearned_size = n.load_network(strLoadFilename);

#ifndef SKIP_TRAINING
	// Train
	n.train(x, y, num_train);
#endif // SKIP_TRAINING

	// Save layers
#ifdef SAVE_FINAL_MN
	if (prelearned_size > 0)
		n.move_network_for_backup(strLoadFilename);
	
	n.save_network(filemgt::get_default_filename());
#endif

	// Testing ====================================
	m.load_testing();
	std::uint32_t num_test = m.test_sample.size();
	std::cout << "Image mapping(Test): " << num_test << std::endl;

	vec2d_t test_x;
	vec_t test_y;
	for (size_t i = 0; i < num_test; i++){
		test_x.push_back(m.test_sample[i]->image);
		test_y.push_back(m.test_sample[i]->label);
	}

    //usleep(1000);
        //Xun 10/07/17: change the testing size, but cannot exceed loaded size in mnist_parser.h
	test_sample_count = 200;
        printf("Testing with %d samples:\n", test_sample_count);
    //    const clock_t begin_time = clock();
    #ifdef GPU
        n.test_single(test_x, test_y, test_sample_count);
    #else
        n.test(test_x, test_y, test_sample_count, 1);
    #endif
    //  cout << "Time consumed in test: " << float(clock() - begin_time) / (CLOCKS_PER_SEC / 1000 ) <<" ms"<<endl;

	return 0;
}
