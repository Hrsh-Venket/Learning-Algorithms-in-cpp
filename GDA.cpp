#include <iostream>
#include <Eigen/Dense>
#include <assert.h>
 
using Eigen::MatrixXd;
 
Eigen::VectorXd PCA(MatrixXd m){
  // Generate Covariance matrix 
  MatrixXd Cov = m/* blah blah*/;
  // find eigenvalues 
  Eigen::EigenSolver<Eigen::MatrixXd> solver(m);
  Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
  // since the covariance matrix is always syymetric, are eigenvalues are never complex
  // find largest eigenvalue
  double largest_eigenvalue = eigenvalues.maxCoeff();
  // find eigenvector corresponding to largest eigenvalue
  assert(largest_eigenvalue==3);
  return eigenvalues;
}

int main()
{
  MatrixXd m(3,3); // by default initialises a matrix with all 0s
  m <<1, 2, 0,
      2, 1, 0, 
      0, 0, 1;
  std::cout << m << "\n" << "The eigenvalues of the matrix are:\n" << PCA(m)  << std::endl;
}