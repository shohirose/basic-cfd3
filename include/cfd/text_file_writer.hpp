#ifndef CFD_TEXT_FILE_WRITER_HPP
#define CFD_TEXT_FILE_WRITER_HPP

#include <Eigen/Core>
#include <filesystem>
#include <string>

namespace cfd {

class TextFileWriter {
 public:
  /**
   * @brief Construct a new Text File Writer object
   *
   * @param directory Directory to output files
   */
  TextFileWriter(const std::filesystem::path& directory)
      : directory_{directory} {}

  /**
   * @brief Write data to a file.
   *
   * @tparam Derived
   * @param x Data
   * @param filename File name
   */
  void write(const Eigen::ArrayXd& x,
             const std::string& filename) const noexcept;

 private:
  std::filesystem::path directory_;  ///> Directory to output files
};

}  // namespace cfd

#endif  // CFD_TEXT_FILE_WRITER_HPP