#pragma once

#include <vector>
#include <cstdint>
#include <time.h>

#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <math.h>
#include <random>
#include <cassert>

#include "settings.h"

// Yeseong: Type controller - also see the kernels.ocl
typedef float float_tt; 
//typedef double float_tt;

namespace convnet {
	typedef std::vector<float_tt> vec_t;
	typedef std::vector<std::vector<float_tt> > vec2d_t;

	inline int uniform_rand(int min, int max) {
		static std::mt19937 gen(0);
		std::uniform_int_distribution<> dst(min, max);
		return dst(gen);
	}

	template<typename T>
	inline T uniform_rand(T min, T max) {
		static std::mt19937 gen(0);
		std::uniform_real_distribution<T> dst(min, max);
		return dst(gen);
	}

	template<typename Iter>
	void uniform_rand(Iter begin, Iter end, float_tt min, float_tt max) {
		for (Iter it = begin; it != end; ++it)
			*it = uniform_rand(min, max);
	}

	void disp_vec_t(vec_t v){
		for (auto i : v)
			std::cout << i << "\t";
		std::cout << "\n";
	}

	void disp_vec2d_t(vec2d_t v){
		for (auto i : v){
			for (auto i_ : i)
				std::cout << i_ << "\t";
			std::cout << "\n";
		}
	}

	float_tt dot(vec_t x, vec_t w){
		assert(x.size() == w.size());
		float_tt sum = 0;
		for (size_t i = 0; i < x.size(); i++){
			sum += x[i] * w[i];
		}
		return sum;
	}

    float_tt dot_per_batch(int batch, vec_t x, vec_t w){
        size_t x_width = w.size();
        float_tt sum = 0;
        for (size_t i = 0; i < x_width; i++){
            sum += x[batch*x_width + i] * w[i];
        }
        return sum;
    }

	struct Image {
		std::vector< std::vector<float_tt> > img;// a image is represented by a 2-dimension vector  
		size_t size; // width or height

		// construction
		Image(size_t size_, std::vector< std::vector<float_tt> > img_) :img(img_), size(size_){}

		// display the image
		void display(){
			for (size_t i = 0; i < size; i++){
				for (size_t j = 0; j < size; j++){
					if (img[i][j] > 200)
						std::cout << 1;
					else
						std::cout << 0;
				}
				std::cout << std::endl;
			}
		}

		// up size to 32, make up with 0
		void upto_32(){
			assert(size < 32);

			std::vector<float_tt> row(32, 0);

			for (size_t i = 0; i < size; i++){
				img[i].insert(img[i].begin(), 0);
				img[i].insert(img[i].begin(), 0);
				img[i].push_back(0);
				img[i].push_back(0);
			}
			img.insert(img.begin(), row);
			img.insert(img.begin(), row);
			img.push_back(row);
			img.push_back(row);

			size = 32;
		}

		std::vector<float_tt> extend(){
			std::vector<float_tt> v;
			for (size_t i = 0; i < size; i++){
				for (size_t j = 0; j < size; j++){
					v.push_back(img[i][j]);
				}
			}
			return v;
		}
	};

	typedef Image* Img;

	struct Sample
	{
		uint8_t label; // label for a specific digit
		std::vector<float_tt> image;
		Sample(float_tt label_, std::vector<float_tt> image_) :label(label_), image(image_){}
	};
} // namespace convnet
