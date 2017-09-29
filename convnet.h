#pragma once
#include "util.h"
#include "convolutional_layer.h"
#include "mnist_parser.h"
#include "layer.h"
#include "maxpooling_layer.h"
#include "output_layer.h"
#include "mnist_parser.h"
#include "fullyconnected_layer.h"
#include "filemgt.hpp"

namespace convnet{
	class ConvNet
	{
	public:
		ConvNet(){}
		~ConvNet(){}

		// Yeseong: Control training mode
		// to check if the back popagration is required in the output layer
		void select_training_mode(bool bInTraining) {
			for (auto layer : layers) {
				layer->bInTraining = bInTraining;
			}
		}

		void train(vec2d_t train_x, vec_t train_y, size_t train_size){
			select_training_mode(true); // Yeseong

#ifdef GPU
            std::cout << "Training with GPU:" << std::endl;
#else
            std::cout << "Training with CPU:" << std::endl;
#endif
			train_x_ = train_x;
			train_y_ = train_y;
			train_size_ = train_size;
			/* 
				auto add OutputLayer as the last layer.
			*/
			//this->add_layer(new OutputLayer(layers.back()->out_depth_)); // Yeseong: Do it in main

			/*
				start training...
			*/
			auto stop = false;
			int iter = 0;
			while (iter < MAX_ITER && !stop){
				iter++;
				auto err = train_once();
				std::cout << iter << ": training cost: " << err << std::endl;
				//if (err < END_CONDITION) stop = true;

				learning_count += M;

#ifdef SAVE_INTER_NN
				if (iter % SAVE_INTER_NN == 0) {
					save_network(filemgt::make_newfilename(learning_count, "-inter")); 
				}
#endif
			}
		}

		// Yeseong Save&Load NN
		void save_network(const std::string& filename) {
			std::ofstream fout(filename);

			filemgt::write_header(fout, layers.size());
			filemgt::write_learning_count(fout, learning_count);

			std::cout << layers.size() << " layers are saving..." << std::endl;

			for (auto layer : layers){
				layer->write_layer(fout);
			}

			std::cout << "File saved to " << filename << std::endl;

			fout.close();
		}

		int load_network(const std::string& filename) {
			if (!filemgt::file_exist(filename)) {
				std::cout << "CNN File not found, so skipped." << std::endl;
				return 0;
			}

			std::ifstream fin(filename);

			size_t layer_size;
			std::string header = filemgt::read_header(fin, layer_size);

			// If needed to check header, DO IT HERE
			std::cout << header << ": " << layer_size << " layers are loading..." << std::endl;
			filemgt::read_learning_count(fin, learning_count);

			int i = 1;
			for (auto layer : layers){
				size_t len_layer = layer->read_layer(fin);
				std::cout << "Layer #" << i << " : " << len_layer << std::endl;
				++i;
			}

			fin.close();

			std::cout << "Loading completed" << std::endl;
			std::cout << "Start from an NN already learned with " << learning_count << " images." << std::endl;
			return learning_count;
		}

		void move_network_for_backup(const std::string& filename) {
			if (!filemgt::file_exist(filename)) {
				std::cout << "File not found (for backup), so skipped" << std::endl;
				return;
			}

			// Retreive fileinfo
			std::ifstream fin(filename);
			
			size_t layer_size;
			int prev_learning_count = 0;
			std::string header = filemgt::read_header(fin, layer_size);
			filemgt::read_learning_count(fin, prev_learning_count);

			// If needed to check header, DO IT HERE

			// Rename for backup
			std::string newfilename = filemgt::make_newfilename(prev_learning_count);
			filemgt::file_rename(filename, newfilename);

			std::cout << "File backuped: " << newfilename << std::endl;
		}

		void test_single(vec2d_t test_x, vec_t test_y, size_t test_size){
			select_training_mode(false); // Yeseong

			test_x_ = test_x, test_y_ = test_y, test_size_ = test_size;
			int iter = 0;
			int bang = 0;

#ifdef GPU
            std::cout << "Testing with GPU " << std::endl;
#else
            std::cout << "Testing with CPU " << std::endl;
#endif // GPU
            while (iter < test_size_){
                int result = 0;
				result=test_once(iter)?1:0;
                bang += result;
                iter++;
				std::cout << iter << " tested" << std::endl;
			}
			std::cout << "bang/test_size_: " << bang << "/" << test_size_ << std::endl;
			std::cout << (int)(bang*100/test_size_) << " %" << std::endl;
		}


		void test(vec2d_t test_x, vec_t test_y, size_t test_size, int batch_size){
			select_training_mode(false); // Yeseong

            assert(batch_size > 0);
            assert(test_size % batch_size == 0);
			test_x_ = test_x, test_y_ = test_y, test_size_ = test_size;
			int iter = 0;
			int bang = 0;

#ifdef GPU
            std::cout << "Testing with batch size of " << batch_size << std::endl;
#else
            std::cout << "Testing with CPU " << std::endl;
#endif // GPU
            while (iter < test_size_ / batch_size){
                int result = 0;
#ifdef GPU // Use GPU
                result = test_once_batch(iter*batch_size, batch_size);
                printf(" Running batch #%d, %d in %d is correct\n", iter, result, batch_size);
    #ifdef CHECK_RESULT     // Check result of batch operations
                bool check = check_batch_result(batch_size);
                if (check)
                    printf("  \\__ Results verified.\n");
    #endif // CHECK_RESULT
#else  // Use CPU
                if (batch_size == 1)
                    result=test_once(iter)?1:0;
                else{
                    std::cout << "Cannot run batch operations with CPU! Abording.." << std::endl;
                    return;
                }
#endif // GPU
                bang += result;
                iter++;
				std::cout << iter * batch_size << " tested" << std::endl;
			}
			std::cout << "bang/test_size_: " << bang << "/" << test_size_ << std::endl;
			std::cout << (int)(bang*100/test_size_) << " %" << std::endl;
		}

        void test(vec2d_t test_x, vec_t test_y, size_t test_size){
            test(test_x, test_y, test_size , 1);
        }

		void add_layer(Layer* layer){
			if (!layers.empty())
				this->layers.back()->next = layer;
			this->layers.push_back(layer);
			layer->next = NULL;
		}

	private:
		size_t max_iter(vec_t v){
			size_t i = 0;
			float_tt max = v[0];
			for (size_t j = 1; j < v.size(); j++){
				if (v[j] > max){
					max = v[j];
					i = j;
				}
			}
			return i;
		}

        size_t max_iter(float_tt v[], size_t size){
            size_t i = 0;
            float_tt max = v[0]; //std::cout<< " raw output: "<<v[0]<<" ";
            for (size_t j = 1; j < size; j++){
                //std::cout<< v[j] << " ";
                if (v[j] > max){
                    max = v[j];
                    i = j;
                }
            }
            //std::cout<<std::endl;
            return i;
        }
        
        bool test_once_random(){
            int test_x_index = uniform_rand(0, test_size_ - 1);
            return test_once(test_x_index);
        }

		bool test_once(int test_x_index){
			layers[0]->input_ = test_x_[test_x_index];
			for (auto layer : layers){
#ifdef GPU
				layer->forward_gpu();
#else
				layer->forward_cpu();
#endif

				if (layer->next != nullptr){
					layer->next->input_ = layer->output_;
				}
			}

			//printf("RET: %d\n", (int)max_iter(layers.back()->output_));

			return (int)test_y_[test_x_index] == (int)max_iter(layers.back()->output_);
		}
        
        int test_once_random_batch(int batch_size) {
            int test_x_index = uniform_rand(0, test_size_ - batch_size);
            return test_once_batch(test_x_index, batch_size);
        }

        int test_once_batch(int test_x_index, int batch_size){
            //std::cout<<"test_x_index: "<<test_x_index<<std::endl;
            layers.back()->exp_y_batch.resize(batch_size);
            // concatenate input vectors into one vector
            for (int s = 0; s < batch_size; s++){
                layers[0]->input_batch_.insert(layers[0]->input_batch_.end(), test_x_[test_x_index + s].begin(), test_x_[test_x_index + s].end());
                layers.back()->exp_y_batch[s] = test_y_[test_x_index + s];
            }
            
            for (auto layer : layers){
                layer->forward_batch(batch_size);
                if (layer->next != nullptr){
                    layer->next->input_batch_ = layer->output_batch_;
                }
            }

            // collect batch results
            int count = 0;
            int outlables = layers.back()->in_depth_;
            for (int sample = 0; sample < batch_size; sample++)
                if ((int)test_y_[test_x_index + sample] == (int)max_iter(&layers.back()->output_batch_[0] + sample*outlables, outlables))
                    count++;
            return count;
        }

        bool check_batch_result(int batch_size) {
            bool all_correct = true;
            for (int sample = 0; sample < batch_size; sample++){
                int each_input_size = layers[0]->in_height_ * layers[0]->in_width_;
                vec_t this_input = vec_t(layers[0]->input_batch_.begin() + sample*each_input_size, 
                                         layers[0]->input_batch_.begin() + (sample + 1)*each_input_size);
                layers[0]->input_ = this_input;
                for (auto layer : layers){
                    layer->forward_cpu();
                    if (layer->next != nullptr){
                        layer->next->input_ = layer->output_;
                    }
                }
                vec_t output_batch = layers.back()->output_batch_;
                vec_t this_output = layers.back()->output_;
                int out_depth = layers.back()->in_depth_;
                for (int out = 0; out < out_depth; out++){
                    //printf("     Checking result of batch #%d out #%d...\n", batch, out);
                    float err = fabs(this_output[out]-output_batch[out + sample*out_depth]);
                    if (err > 5e-3){
                        printf("   !!==Wrong output. Sample: #%d, Out: #%d, should be: %f, batch result: %f\n",
                            sample, out, this_output[out], output_batch[out + sample*out_depth]);
                        all_correct = false;
                    }
                }
                int outlables = layers.back()->in_depth_;
                int cpu_result = (int)max_iter(layers.back()->output_);
                int gpu_result = (int)max_iter(&layers.back()->output_batch_[0] + sample*outlables, outlables);
                if ( cpu_result != gpu_result)
                    printf("result #%d mismatch: CPU output: %d, GPU output: %d\n", sample, cpu_result, gpu_result);
            }
            return all_correct;
        }

		float_tt train_once(){
			float_tt err = 0;
			int iter = 0;
			while (iter < M){
                //auto train_x_index = iter % train_size_;
				iter++;
				auto train_x_index = uniform_rand(0, train_size_ - 1);
				layers[0]->input_ = train_x_[train_x_index];
				layers.back()->exp_y = (int)train_y_[train_x_index];
				/*
				Start forward feeding.
				*/

				//std::cout << layers.size() << std::endl;

				for (auto layer : layers){
#ifdef GPU
                    layer->forward_gpu();
#else
					layer->forward_cpu();
#endif
					if (layer->next != nullptr){
						layer->next->input_ = layer->output_;
					}
				}
				err += layers.back()->err;
				/*
				back propgation
				*/
				
				for (auto i = layers.rbegin(); i != layers.rend(); i++){
					(*i)->back_prop();
				}
			}
			return err / M;
		}

		std::vector < Layer* > layers;
		
		size_t train_size_;
		vec2d_t train_x_;
		vec_t train_y_;

		size_t test_size_;
		vec2d_t test_x_;
		vec_t test_y_;

		// Yeseong
		int learning_count = 0;
	};
} //namespace convnet
