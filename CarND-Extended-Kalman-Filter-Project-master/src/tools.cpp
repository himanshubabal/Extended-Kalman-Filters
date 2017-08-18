#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size()
       || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){

        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //check division by zero
    if (px == 0 && py == 0){
        throw string("CalculateJacobian() - Error - Division by Zero");
    } else {
        float px2 = pow(px, 2);
        float py2 = pow(py, 2);
        float px2_py2 = px2 + py2;
        float sqrt_px2_py2 = sqrt(px2_py2);
        float sqrt_px2_py2_cube = pow(sqrt_px2_py2, 3);

        float _0_0 = px / sqrt_px2_py2;
        float _1_0 = - (py / px2_py2);
        float _2_0 = (py * (vx * py - vy * px))/sqrt_px2_py2_cube;

        float _0_1 = py / sqrt_px2_py2;
        float _1_1 = px / px2_py2;
        float _2_1 = (px * (vy * px - vx * py))/sqrt_px2_py2_cube;

        float _2_2 = px / sqrt_px2_py2;
        float _2_3 = py / sqrt_px2_py2;

        Hj << _0_0, _0_1, 0, 0,
                _1_0, _1_1, 0, 0,
                _2_0, _2_1, _2_2, _2_3;
    }

    //compute the Jacobian matrix

    return Hj;
}
