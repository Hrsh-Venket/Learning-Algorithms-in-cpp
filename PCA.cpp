// due to issues setting up eigen path, run with:
// g++ PCA.cpp -o test -I /usr/include/eigen3
#include <chrono>
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <assert.h>
#include <fstream>
 
using Eigen::MatrixXd;


using namespace Eigen;

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}


Eigen::VectorXd PCA(MatrixXd m){
  // time measurement


  // Generate Covariance matrix 
  Eigen::MatrixXd centered = m.rowwise() - m.colwise().mean(); // subtract column means from each row
  Eigen::MatrixXd Cov = (1.0/(m.rows()-1)) * centered.adjoint() * centered; // calculate covariance matrix
  assert(Cov.rows() == Cov.cols() && "Covariance matrix is not square"); // this is a check
  // find eigenvalues 
  Eigen::EigenSolver<Eigen::MatrixXd> solver(Cov);
  Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
  // since the covariance matrix is always syymetric, are eigenvalues are never complex
  // find largest eigenvalue
  double largest_eigenvalue = eigenvalues.maxCoeff();
  // find eigenvector corresponding to largest eigenvalue
  Eigen::MatrixXcd eigenvectors = solver.eigenvectors(); // find all eigenvectors
  int largest_eigenvalue_index;
  for (int i=0; i<eigenvalues.size(); i++) {
    if (eigenvalues[i] == largest_eigenvalue) { // find the corresponding eigenvector's index
      largest_eigenvalue_index = i;
      break;
    }
  }
  Eigen::VectorXd principal_component = eigenvectors.col(largest_eigenvalue_index).real(); 
  return principal_component;
}

int main()
{
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  MatrixXd m = load_csv<MatrixXd>("massivedataset.csv");

  auto t1 = high_resolution_clock::now();
  Eigen::VectorXd output = PCA(m);
  auto t2 = high_resolution_clock::now();
  

  /* Getting number of milliseconds as an integer. */
  auto ms_int = duration_cast<milliseconds>(t2 - t1);

  /* Getting number of milliseconds as a double. */
  duration<double, std::milli> ms_double = t2 - t1;
  std::cout << "time taken is\n";
  // std::cout << ms_int.count() << "ms\n";
  std::cout << ms_double.count() << "ms\n";
  std::cout << "answer is \n";
  std::cout  << output  << std::endl;
}