//
// Created by qiang on 2022/3/18.
//
#include <iostream>
#include <cmath>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

int main(int argc, char **argv) {
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    Eigen::AngleAxisd rotation_vector(M_PI / 4, Eigen::Vector3d(0, 0, 1));
    std::cout.precision(3);
    std::cout << "rotation matrix = \n" << rotation_vector.matrix() << std::endl;

    rotation_matrix = rotation_vector.toRotationMatrix();

    Eigen::Vector3d v(1, 0, 0);
    Eigen::Vector3d v_rotated = rotation_vector * v;
    std::cout << "(1,0,0) after rotation = " << v_rotated.transpose() << std::endl;

    v_rotated = rotation_matrix * v;
    std::cout << "(1,0,0) after rotation = " << v_rotated.transpose() << std::endl;

    Eigen::Vector3d euler_angles = rotation_matrix.eulerAngles(2,1,0);
    std::cout << "yaw pitch roll = "<< euler_angles.transpose() << std::endl;

    Eigen::Isometry3d T=Eigen::Isometry3d ::Identity();
    T.rotate(rotation_vector);
    T.pretranslate(Eigen::Vector3d(1,3,4));
    std::cout << "Transform matrix = \n" << T.matrix() << std::endl;

    Eigen::Vector3d v_transformed = T*v;
    std::cout << "v transformed = " <<v_transformed.transpose() << std::endl;

    Eigen::Quaterniond q=Eigen::Quaterniond (rotation_vector);
    std::cout << "quaternion = \n" << q.coeffs()<< std::endl;

    q = Eigen::Quaterniond (rotation_matrix);
    std::cout << "quaternion = \n" << q.coeffs() << std::endl;

    v_rotated = q*v;
    std::cout<< "(1,0,0) after rotation = " << v_rotated.transpose() << std::endl;

    return 0;
}
