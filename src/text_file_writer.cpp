#include "cfd/text_file_writer.hpp"

#include <fstream>
#include <iostream>

namespace cfd {

void TextFileWriter::write(const Eigen::ArrayXd& x,
                           const std::string& filename) const noexcept {
  namespace fs = std::filesystem;
  if (!fs::exists(directory_)) {
    std::error_code ec;
    fs::create_directories(directory_, ec);
    if (ec) {
      std::cerr << "Failed to create a directory: " << directory_
                << "\nError code: " << ec.message() << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
  std::ofstream file(directory_ / fs::path(filename));
  file << x << std::endl;
}

}  // namespace cfd