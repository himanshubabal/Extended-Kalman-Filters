#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include "Eigen/Dense"
#include "measurement_package.h"
#include "tracking.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;


int main() {

    /*******************************************************************************
     *  Set Measurements															 *
     *******************************************************************************/
    vector<MeasurementPackage> measurement_pack_list;

    // hardcoded input file with laser and radar measurements
    string dir_path = "/Users/himanshubabal/Desktop/SDC/Kalman_Filters/cpp/";
    string in_file_name_ = dir_path + "obj_pose-laser-radar-synthetic-input.txt";

    ifstream in_file(in_file_name_.c_str(),std::ifstream::in);

    if (!in_file.is_open()) {
        cout << "Cannot open input file: " << in_file_name_ << endl;
    }

    string line;
    // set i to get only first 3 measurments
    int i = 0;
    while(getline(in_file, line) && (i<=3)){
//    while(getline(in_file, line)){

        MeasurementPackage meas_package;

        istringstream iss(line);
        string sensor_type;
        iss >> sensor_type;	//reads first element from the current line
        long timestamp;
        if(sensor_type.compare("L") == 0){	//laser measurement
            //read measurements
            meas_package.sensor_type_ = MeasurementPackage::LASER;
            meas_package.raw_measurements_ = VectorXd(2);
            float x;
            float y;
            iss >> x;
            iss >> y;
            meas_package.raw_measurements_ << x,y;
            iss >> timestamp;
            meas_package.timestamp_ = timestamp;
            measurement_pack_list.push_back(meas_package);

        }else if(sensor_type.compare("R") == 0){
            //Skip Radar measurements
            continue;
        }
        i++;

    }

    //Create a Tracking instance
    Tracking tracking;

    //call the ProcessingMeasurement() function for each measurement
    size_t N = measurement_pack_list.size();
    for (size_t k = 0; k < N; ++k) {	//start filtering from the second frame (the speed is unknown in the first frame)
        tracking.ProcessMeasurement(measurement_pack_list[k]);

    }

    if(in_file.is_open()){
        in_file.close();
    }
    return 0;
}

MatrixXd CalculateJacobian(const VectorXd& x_state) {

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

VectorXd CalculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truth){

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
