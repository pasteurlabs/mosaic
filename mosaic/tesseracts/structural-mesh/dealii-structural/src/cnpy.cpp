// mosaic:io
// cnpy — C++ library for reading/writing NumPy .npy files.
// MIT License — original by Carl Rogers (https://github.com/rogersce/cnpy).

#include "cnpy.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <complex>
#include <cstring>
#include <cassert>
#include <regex>
#include <typeinfo>

namespace cnpy {

// ---------------------------------------------------------------------------
// Type-map: C++ type → NumPy dtype string
// ---------------------------------------------------------------------------
// mosaic:util
static char BigEndianChar() {
  int x = 1;
  return (*(char*)&x == 1) ? '<' : '>';
}

template<typename T> struct TypeStr { static std::string str() { return "V1"; } };
template<> struct TypeStr<float>    { static std::string str() {
  std::string s; s += BigEndianChar(); s += "f4"; return s; } };
template<> struct TypeStr<double>   { static std::string str() {
  std::string s; s += BigEndianChar(); s += "f8"; return s; } };
template<> struct TypeStr<int32_t>  { static std::string str() {
  std::string s; s += BigEndianChar(); s += "i4"; return s; } };
template<> struct TypeStr<int64_t>  { static std::string str() {
  std::string s; s += BigEndianChar(); s += "i8"; return s; } };
template<> struct TypeStr<uint32_t> { static std::string str() {
  std::string s; s += BigEndianChar(); s += "u4"; return s; } };
template<> struct TypeStr<uint64_t> { static std::string str() {
  std::string s; s += BigEndianChar(); s += "u8"; return s; } };

// ---------------------------------------------------------------------------
// Build .npy header
// ---------------------------------------------------------------------------
// mosaic:io
static std::vector<char> build_npy_header(
  const std::string &dtype_str,
  bool fortran_order,
  const std::vector<size_t> &shape)
{
  std::ostringstream dict;
  dict << "{'descr': '" << dtype_str << "', 'fortran_order': "
       << (fortran_order ? "True" : "False") << ", 'shape': (";
  for (size_t i = 0; i < shape.size(); ++i) {
    dict << shape[i];
    if (i + 1 < shape.size() || shape.size() == 1) dict << ",";
  }
  dict << "), }";

  std::string dict_str = dict.str();
  // Pad to multiple of 64 bytes (total header = 10 + dict_str + \n)
  size_t header_len_raw = 10 + dict_str.size() + 1; // magic(6)+major(1)+minor(1)+hlen(2)+dict+\n
  size_t pad = (64 - (header_len_raw % 64)) % 64;
  dict_str.append(pad, ' ');
  dict_str += '\n';

  uint16_t header_data_len = (uint16_t)dict_str.size();

  std::vector<char> header;
  header.push_back('\x93');
  header.push_back('N');
  header.push_back('U');
  header.push_back('M');
  header.push_back('P');
  header.push_back('Y');
  header.push_back('\x01'); // major version
  header.push_back('\x00'); // minor version
  // Little-endian 2-byte header data length
  header.push_back((char)(header_data_len & 0xFF));
  header.push_back((char)((header_data_len >> 8) & 0xFF));
  for (char c : dict_str) header.push_back(c);

  return header;
}

// ---------------------------------------------------------------------------
// Parse .npy header
// ---------------------------------------------------------------------------
// mosaic:io
static void parse_npy_header(std::istream &is,
                              size_t &word_size,
                              std::vector<size_t> &shape,
                              bool &fortran_order)
{
  // Magic: \x93NUMPY
  char magic[6];
  is.read(magic, 6);
  if (is.fail() || magic[0] != '\x93' ||
      std::string(magic+1, 5) != "NUMPY")
    throw std::runtime_error("cnpy: not a valid .npy file");

  uint8_t major, minor;
  is.read((char*)&major, 1);
  is.read((char*)&minor, 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t hlen16 = 0;
    is.read((char*)&hlen16, 2);
    header_len = hlen16;
  } else {
    is.read((char*)&header_len, 4);
  }

  std::string header_str(header_len, ' ');
  is.read(&header_str[0], header_len);

  // Parse 'descr'
  {
    auto find = [&](const std::string &key) -> std::string {
      size_t pos = header_str.find(key);
      if (pos == std::string::npos) return "";
      pos = header_str.find("'", pos + key.size());
      if (pos == std::string::npos) return "";
      size_t start = pos + 1;
      size_t end   = header_str.find("'", start);
      if (end == std::string::npos) return "";
      return header_str.substr(start, end - start);
    };

    std::string descr = find("'descr'");
    // Determine word size from dtype string
    if (descr.size() >= 2) {
      char sz = descr.back();
      word_size = sz - '0';
    } else {
      word_size = 4;
    }
  }

  // Parse 'fortran_order'
  {
    size_t pos = header_str.find("'fortran_order'");
    if (pos != std::string::npos) {
      pos = header_str.find(":", pos);
      if (pos != std::string::npos) {
        // Skip whitespace
        while (pos < header_str.size() && std::isspace(header_str[pos])) ++pos;
        ++pos; // skip ':'
        while (pos < header_str.size() && std::isspace(header_str[pos])) ++pos;
        fortran_order = (header_str.substr(pos, 4) == "True");
      }
    } else {
      fortran_order = false;
    }
  }

  // Parse 'shape'
  {
    size_t pos = header_str.find("'shape'");
    if (pos == std::string::npos) pos = header_str.find("\"shape\"");
    if (pos != std::string::npos) {
      size_t lp = header_str.find("(", pos);
      size_t rp = header_str.find(")", pos);
      if (lp != std::string::npos && rp != std::string::npos) {
        std::string shape_str = header_str.substr(lp+1, rp-lp-1);
        shape.clear();
        std::istringstream ss(shape_str);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
          // trim
          tok.erase(0, tok.find_first_not_of(" \t\n\r"));
          tok.erase(tok.find_last_not_of(" \t\n\r") + 1);
          if (!tok.empty())
            shape.push_back((size_t)std::stoull(tok));
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// npy_load
// ---------------------------------------------------------------------------
// mosaic:io
NpyArray npy_load(const std::string &fname)
{
  std::ifstream fs(fname, std::ios::binary);
  if (!fs.is_open())
    throw std::runtime_error("cnpy: cannot open file: " + fname);

  size_t word_size = 0;
  std::vector<size_t> shape;
  bool fortran_order = false;
  parse_npy_header(fs, word_size, shape, fortran_order);

  NpyArray arr(shape, word_size, fortran_order);
  fs.read(arr.data<char>(), (std::streamsize)(arr.num_vals() * word_size));
  if (fs.fail())
    throw std::runtime_error("cnpy: failed to read data from: " + fname);
  return arr;
}

// ---------------------------------------------------------------------------
// npy_save (template specialisations instantiated here)
// ---------------------------------------------------------------------------
// mosaic:io
template<typename T>
void npy_save(const std::string &fname, const T *data,
              const std::vector<size_t> &shape,
              const std::string &mode)
{
  std::string dtype = TypeStr<T>::str();
  std::vector<char> header = build_npy_header(dtype, false, shape);

  size_t n = 1;
  for (size_t s : shape) n *= s;

  auto open_mode = (mode == "a")
    ? (std::ios::binary | std::ios::app)
    : (std::ios::binary | std::ios::out | std::ios::trunc);

  std::ofstream fs(fname, open_mode);
  if (!fs.is_open())
    throw std::runtime_error("cnpy: cannot open file for writing: " + fname);

  fs.write(header.data(), (std::streamsize)header.size());
  fs.write(reinterpret_cast<const char*>(data), (std::streamsize)(n * sizeof(T)));
}

// Explicit instantiations
template void npy_save<float>   (const std::string&, const float*,    const std::vector<size_t>&, const std::string&);
template void npy_save<double>  (const std::string&, const double*,   const std::vector<size_t>&, const std::string&);
template void npy_save<int32_t> (const std::string&, const int32_t*,  const std::vector<size_t>&, const std::string&);
template void npy_save<int64_t> (const std::string&, const int64_t*,  const std::vector<size_t>&, const std::string&);
template void npy_save<uint32_t>(const std::string&, const uint32_t*, const std::vector<size_t>&, const std::string&);
template void npy_save<uint64_t>(const std::string&, const uint64_t*, const std::vector<size_t>&, const std::string&);

} // namespace cnpy
