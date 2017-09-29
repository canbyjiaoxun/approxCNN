#pragma once

#include "util.h"
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// Yeseong: Save&Load NN
namespace filemgt {
	const char* MAGIC = "APPROXCNN-V1.0######"; // Must be 20 bytes

	template<typename T>
	void write_ptype(std::ofstream& fout, T data) { // Write primitive type
		fout.write(reinterpret_cast<char*>(&data), sizeof(T));
	}

	template<typename T>
	void read_ptype(std::ifstream& fin, T& data) { // Read primitive type
		fin.read(reinterpret_cast<char*>(&data), sizeof(T));
	}

	void write_header(std::ofstream& fout, size_t layer_size) {
		fout.write(MAGIC, strlen(MAGIC));
		write_ptype(fout, layer_size);
	}

	std::string read_header(std::ifstream& fin, size_t& layer_size) {
		size_t magic_size = strlen(MAGIC);
		std::string loaded_magic(magic_size+1, '\0');

		fin.read(&loaded_magic[0], magic_size);
		read_ptype(fin, layer_size);

		return loaded_magic;
	}

	void write_learning_count(std::ofstream& fout, int learning_count) {
		write_ptype(fout, learning_count);
	}

	void read_learning_count(std::ifstream& fin, int& learning_count) {
		read_ptype(fin, learning_count);
	}


	void write_vector(std::ofstream& fout, const convnet::vec_t& vec) {
		size_t vec_size = vec.size();
		write_ptype(fout, vec_size);

		for (float_tt d : vec) {
			if (sizeof(float_tt) != sizeof(float)) {
				float t = (float) d;
				write_ptype(fout, t);
			} else {
				write_ptype(fout, d);
			}
		}
	}

	size_t read_vector(std::ifstream& fin, convnet::vec_t& vec) {
		size_t vec_size;
		read_ptype(fin, vec_size);

		assert(vec_size == vec.size());
		vec.clear();

		for (int i = 0; i < vec_size; ++i) {
			float d;
			read_ptype(fin, d);
			vec.push_back((float_tt)d);
		}

		return vec_size;
	}

	bool file_exist(const std::string& filename) {
		struct stat buffer;   
		return (stat (filename.c_str(), &buffer) == 0); 
	}

	bool file_rename(const std::string& oldfilename, const std::string& newfilename) {
		return rename(oldfilename.c_str(), newfilename.c_str()) == 0;
	}

	std::string make_newfilename(int learning_count, const std::string postfix = "") {
		std::stringstream newname;
		newname << NN_DATA_FILENAME << postfix << "." << learning_count << "." << NN_DATA_FILENAME_EXT;
		return newname.str();
	}

	std::string get_default_filename() {
		std::stringstream newname;
		newname << NN_DATA_FILENAME << "." << NN_DATA_FILENAME_EXT;
		return newname.str();
	}
}
