// cnpy — C++ library for reading/writing NumPy .npy and .npz files.
// MIT License — original by Carl Rogers (https://github.com/rogersce/cnpy).
// This is a self-contained single-header/implementation split version.
// mosaic:util  (third-party I/O utility — not authored solver code)

#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <cassert>
#include <typeinfo>

namespace cnpy {

// mosaic:io
struct NpyArray {
  NpyArray(const std::vector<size_t> &shape, size_t word_size_, bool fortran_order_)
    : shape(shape), word_size(word_size_), fortran_order(fortran_order_)
  {
    num_vals_ = 1;
    for (size_t s : shape) num_vals_ *= s;
    data_holder = std::shared_ptr<std::vector<char>>(
      new std::vector<char>(num_vals_ * word_size_));
  }

  NpyArray() : shape(0), word_size(0), fortran_order(false), num_vals_(0) {}

  template<typename T>
  T* data() {
    return reinterpret_cast<T*>(&(*data_holder)[0]);
  }

  template<typename T>
  const T* data() const {
    return reinterpret_cast<const T*>(&(*data_holder)[0]);
  }

  size_t num_vals() const { return num_vals_; }

  std::shared_ptr<std::vector<char>> data_holder;
  std::vector<size_t> shape;
  size_t word_size;
  bool fortran_order;
  size_t num_vals_;
};

// mosaic:io
NpyArray npy_load(const std::string &fname);

template<typename T>
void npy_save(const std::string &fname, const T *data,
              const std::vector<size_t> &shape,
              const std::string &mode = "w");

} // namespace cnpy
