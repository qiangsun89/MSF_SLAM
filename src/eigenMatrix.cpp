//
// Created by qiang on 2022/3/17.
//

#include <iostream>
#include <ctime>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>


int main(int argc, char **argv){
    Eigen::Matrix<float, 2,3> matrix_23;
    Eigen::Vector3d v_3d;
    Eigen::Matrix<float, 3, 1> vd_3d;

    Eigen::Matrix3d matrix_33 = Eigen::Matrix3d::Zero();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> matrix_dynamic;
    Eigen::MatrixXd matrix_x;

    matrix_23 << 1,2,3,4,5,6;
    std::cout << matrix_23 << std::endl;

    for(int i =0; i<2; i++){
        for(int j =0; j<3; j++){
            std::cout << matrix_23(i,j) << "\t";
        }
        std::cout<<std::endl;
    }

    v_3d << 3, 2, 1;
    vd_3d << 4, 5, 6;
    Eigen::Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
    std::cout << result << std::endl;

    Eigen::Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
    std::cout << result2 << std::endl;

    matrix_33 =Eigen::Matrix3d::Random();
    std::cout << matrix_33 << std::endl;
    std::cout<<"==========="<< std::endl;
    std::cout << matrix_33.transpose() << std::endl;
    std::cout<<"==========="<< std::endl;
    std::cout << matrix_33.sum() << std::endl;
    std::cout<<"==========="<< std::endl;
    std::cout << matrix_33.trace() << std::endl;
    std::cout<<"==========="<< std::endl;
    std::cout << 10*matrix_33 << std::endl;
    std::cout<<"==========="<< std::endl;
    std::cout << matrix_33.inverse() << std::endl;
    std::cout<<"==========="<< std::endl;
    std::cout << matrix_33.determinant() << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen_solver(matrix_33.transpose()*matrix_33);
    std::cout << "Eigen values = \n" << eigen_solver.eigenvalues()<< std::endl;
    std::cout << "Eigen vectors= \n" << eigen_solver.eigenvectors() << std::endl;

    Eigen::Matrix< double, 50, 50 > matrix_NN;
    matrix_NN = Eigen::MatrixXd::Random(50, 50);
    Eigen::Matrix<double, 50,1> v_Nd;
    v_Nd = Eigen::MatrixXd::Random(50,1);
    clock_t  time_tt = clock();
    Eigen::Matrix<double, 50, 1> x=matrix_NN.inverse() *v_Nd;
    std::cout << "time use in normal inverse is " << 1000*(clock()-time_tt)/(double )CLOCKS_PER_SEC << "ms"<< std::endl;

    clock_t time_stt = clock();
    Eigen::Matrix<double, 50, 1> xx=matrix_NN.colPivHouseholderQr().solve(v_Nd);
    std::cout << "time use in QR decomposition is "<< 1000*(clock()-time_stt)/(double )CLOCKS_PER_SEC << "ms"<< std::endl;
    return 0;
}

