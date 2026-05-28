// cnpy — NumPy .npy / .npz I/O for C++
// Original: https://github.com/rogersce/cnpy  (MIT licence)
// Vendored and trimmed: only npy_save / npy_load used by heat_solver.
//
// MIT License
// Copyright (c) 2011 Carl Rogers
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.

#pragma once

#include <cassert>
#include <complex>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace cnpy {

// ---------------------------------------------------------------------------
// NpyArray — holds a loaded array
// ---------------------------------------------------------------------------
struct NpyArray {
    NpyArray(const std::vector<size_t>& _shape, size_t _word_size,
             bool _fortran_order)
        : shape(_shape), word_size(_word_size), fortran_order(_fortran_order),
          num_vals(1) {
        for (size_t s : shape) num_vals *= s;
        data_holder = std::shared_ptr<std::vector<char>>(
            new std::vector<char>(num_vals * word_size));
    }
    NpyArray() : shape(0), word_size(0), fortran_order(false), num_vals(0) {}

    template <typename T> T* data() {
        return reinterpret_cast<T*>(&(*data_holder)[0]);
    }
    template <typename T> const T* data() const {
        return reinterpret_cast<const T*>(&(*data_holder)[0]);
    }
    template <typename T> std::vector<T> as_vec() const {
        const T* p = data<T>();
        return std::vector<T>(p, p + num_vals);
    }
    size_t num_bytes() const { return num_vals * word_size; }

    std::shared_ptr<std::vector<char>> data_holder;
    std::vector<size_t> shape;
    size_t word_size;
    bool fortran_order;
    size_t num_vals;
};

// ---------------------------------------------------------------------------
// Type → numpy dtype string
// ---------------------------------------------------------------------------
template <typename T> std::string dtype_str() {
    // little-endian prefix '<' is default for all numeric types
    if (std::is_same<T, float>::value)  return "<f4";
    if (std::is_same<T, double>::value) return "<f8";
    if (std::is_same<T, int32_t>::value)  return "<i4";
    if (std::is_same<T, int64_t>::value)  return "<i8";
    if (std::is_same<T, uint32_t>::value) return "<u4";
    if (std::is_same<T, uint64_t>::value) return "<u8";
    throw std::runtime_error("cnpy: unsupported type");
}

// ---------------------------------------------------------------------------
// Build the .npy header string
// ---------------------------------------------------------------------------
inline std::string build_npy_header(const std::string& dtype,
                                    const std::vector<size_t>& shape,
                                    bool fortran_order = false) {
    std::string dict = "{'descr': '" + dtype + "', 'fortran_order': ";
    dict += fortran_order ? "True" : "False";
    dict += ", 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) {
        dict += std::to_string(shape[i]);
        if (i + 1 < shape.size()) dict += ", ";
    }
    if (shape.size() == 1) dict += ",";
    dict += "), }";

    // Header must be padded to a multiple of 64 bytes (npy v1.0 spec).
    // Magic (6) + major (1) + minor (1) + header_len (2) = 10 bytes prefix.
    size_t prefix_len = 10;
    // Total = prefix_len + len(dict) + padding + 1 ('\n')
    // Must be multiple of 64.
    size_t dict_len = dict.size() + 1; // +1 for '\n'
    size_t total = prefix_len + dict_len;
    size_t padding = (64 - total % 64) % 64;
    dict.append(padding, ' ');
    dict += '\n';

    uint16_t header_len = static_cast<uint16_t>(dict.size());
    std::string header;
    header += '\x93';
    header += "NUMPY";
    header += '\x01'; // major version
    header += '\x00'; // minor version
    // little-endian uint16
    header += static_cast<char>(header_len & 0xFF);
    header += static_cast<char>((header_len >> 8) & 0xFF);
    header += dict;
    return header;
}

// ---------------------------------------------------------------------------
// npy_save — write a raw C array as .npy
// ---------------------------------------------------------------------------

template <typename T>
void npy_save(const std::string& fname, const T* data,
              const std::vector<size_t>& shape,
              const std::string& mode = "w") {
    std::string dtype = dtype_str<T>();
    std::string header = build_npy_header(dtype, shape);

    std::ofstream out(fname, std::ios::binary | (mode == "a" ? std::ios::app : std::ios::trunc));
    if (!out.is_open())
        throw std::runtime_error("cnpy::npy_save: cannot open file " + fname);
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    size_t n = 1;
    for (size_t s : shape) n *= s;
    out.write(reinterpret_cast<const char*>(data),
              static_cast<std::streamsize>(n * sizeof(T)));
}

template <typename T>
void npy_save(const std::string& fname, const std::vector<T>& data,
              const std::vector<size_t>& shape,
              const std::string& mode = "w") {
    npy_save(fname, data.data(), shape, mode);
}

// ---------------------------------------------------------------------------
// npy_load — load a .npy file
// ---------------------------------------------------------------------------

inline NpyArray npy_load(const std::string& fname) {
    std::ifstream in(fname, std::ios::binary);
    if (!in.is_open())
        throw std::runtime_error("cnpy::npy_load: cannot open file " + fname);

    // Check magic
    char magic[6];
    in.read(magic, 6);
    if (std::string(magic, 6) != "\x93NUMPY")
        throw std::runtime_error("cnpy::npy_load: not a npy file");

    uint8_t major, minor;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);

    uint32_t header_len;
    if (major == 1) {
        uint16_t hl;
        in.read(reinterpret_cast<char*>(&hl), 2);
        header_len = hl;
    } else if (major == 2) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        throw std::runtime_error("cnpy::npy_load: unsupported npy version");
    }

    std::string header(header_len, ' ');
    in.read(&header[0], header_len);

    // Parse fortran_order
    bool fortran_order = header.find("'fortran_order': True") != std::string::npos;

    // Parse shape — find tuple after 'shape':
    size_t pos = header.find("'shape':");
    if (pos == std::string::npos)
        throw std::runtime_error("cnpy::npy_load: no 'shape' in header");
    pos = header.find('(', pos);
    size_t end = header.find(')', pos);
    std::string shape_str = header.substr(pos + 1, end - pos - 1);
    std::vector<size_t> shape;
    std::stringstream ss(shape_str);
    std::string token;
    while (std::getline(ss, token, ',')) {
        // trim whitespace
        size_t s = token.find_first_not_of(" \t");
        if (s == std::string::npos) continue;
        token = token.substr(s);
        s = token.find_last_not_of(" \t");
        if (s != std::string::npos) token = token.substr(0, s + 1);
        if (!token.empty()) shape.push_back(std::stoul(token));
    }
    if (shape.empty()) shape.push_back(0); // 0-d scalar

    // Parse descr (dtype) to get word_size
    size_t d = header.find("'descr':");
    if (d == std::string::npos)
        throw std::runtime_error("cnpy::npy_load: no 'descr' in header");
    size_t q1 = header.find('\'', d + 8);
    size_t q2 = header.find('\'', q1 + 1);
    std::string descr = header.substr(q1 + 1, q2 - q1 - 1);
    // word size is the last character (or characters) as a number
    size_t word_size = std::stoul(descr.substr(2));

    NpyArray arr(shape, word_size, fortran_order);
    in.read(arr.data<char>(), static_cast<std::streamsize>(arr.num_bytes()));
    return arr;
}

} // namespace cnpy
