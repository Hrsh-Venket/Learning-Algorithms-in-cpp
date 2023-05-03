// due to issues setting up eigen path, run with:
// g++ GDA.cpp -o test -I /usr/include/eigen3

#include <iostream>
#include <Eigen/Dense>
#include <assert.h>
 
using Eigen::MatrixXd;


Eigen::VectorXd PCA(MatrixXd m){
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
  MatrixXd m(2, 2);
  m << 1, 0, 
       0, 1;
  std::cout  << PCA(m)  << std::endl;
}