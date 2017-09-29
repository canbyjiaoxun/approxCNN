#include "util.h"
#include "filemgt.hpp"

#pragma once
namespace convnet{
	class Layer
	{
	public:
		Layer(size_t in_width, size_t in_height, size_t in_depth,
			size_t out_width, size_t out_height, size_t out_depth, float_tt alpha, float_tt lambda) :
			in_width_(in_width), in_height_(in_height), in_depth_(in_depth),
			out_width_(out_width), out_height_(out_height), out_depth_(out_depth),
			alpha_(alpha), lambda_(lambda), bInTraining(true)
		{}

		virtual void init_weight() = 0;
		virtual void forward_cpu() = 0;
        virtual void forward_batch(int batch_size) = 0;
		virtual void back_prop() = 0;

        virtual void forward_gpu(){ forward_cpu(); }

		float_tt sigmod(float_tt in){
			return 1.0 / (1.0 + std::exp(-in));
		}


		// Approximate one to avoid use of the exponential function
		// which generates unsupported instructions on multi2sim
		float_tt sigmod2(float_tt value){
			float_tt x = (value < 0)? -value:value;
			float_tt x2 = x*x;
			float_tt e = 1.0f + x + x2*0.555f + x2*x2*0.143f;
			return 1.0f / (1.0f + (value > 0 ? 1.0f / e : e));
		}

		float_tt df_sigmod(float_tt f_x) {
			return f_x * (1.0 - f_x);
		}

		size_t fan_in(){
			return in_width_ * in_height_ * in_depth_;
		}

		size_t fan_out(){
			return out_width_ * out_height_ * out_height_;
		}

		size_t in_width_;
		size_t in_height_;
		size_t in_depth_;

		size_t out_width_;
		size_t out_height_;
		size_t out_depth_;

		vec_t W_;
		vec_t b_;

		// Yeseong: Save&Load NN
		void write_layer(std::ofstream& fout) {
			filemgt::write_vector(fout, W_); // weight
			filemgt::write_vector(fout, b_); // back prop
		}

		size_t read_layer(std::ifstream& fin) {
			size_t len = filemgt::read_vector(fin, W_); // weight
			len += filemgt::read_vector(fin, b_); // back prop
			return len;
		}

		vec_t deltaW_;

		vec_t input_;
		vec_t output_;

        vec_t input_batch_;
        vec_t output_batch_;

		Layer* next;

		float_tt alpha_; // learning rate
		float_tt lambda_; // momentum
		vec_t g_; // err terms

		/*output*/
		float_tt err;
		int exp_y;
		vec_t exp_y_vec;

        vec_t exp_y_batch;
        vec_t exp_y_vec_batch;

		// Yeseong: training mode selection
		bool bInTraining;
	};
}
