#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
    is_initialized_ = false;

    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;

    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;

    // initial state vector
    x_ = VectorXd(5);
    // You won't know where the bicycle is until you receive the first sensor measurement.
    // Once the first sensor measurement arrives, you can initialize px and p​y
    // For the other variables in the state vector x, you can try different initialization
    // values to see what works best.

    // initial covariance matrix
    P_ = MatrixXd(5, 5);

//    // Process noise standard deviation longitudinal acceleration in m/s^2
//    std_a_ = 30;
//
//    // Process noise standard deviation yaw acceleration in rad/s^2
//    std_yawdd_ = 30;
//
//    // Laser measurement noise standard deviation position1 in m
//    std_laspx_ = 0.15;
//
//    // Laser measurement noise standard deviation position2 in m
//    std_laspy_ = 0.15;
//
//    // Radar measurement noise standard deviation radius in m
//    std_radr_ = 0.3;
//
//    // Radar measurement noise standard deviation angle in rad
//    std_radphi_ = 0.03;
//
//    // Radar measurement noise standard deviation radius change in m/s
//    std_radrd_ = 0.3;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.5;
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.57;
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    /**
    Complete the initialization. See ukf.h for other member properties.
    Hint: one or more values initialized above might be wildly off...
    */

    n_x_ = x_.size();
    n_aug_ = n_x_ + 2;
    n_sigma_ = 2 * n_aug_ + 1;

    Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

    lambda_ = 3 - n_aug_;

    weights_ = VectorXd(n_sigma_);

    R_Radar_ = MatrixXd(3, 3);
    R_Radar_ << std_radr_ * std_radr_,  0,                          0,
                0,                      std_radphi_ * std_radphi_,  0,
                0,                      0,                          std_radrd_ * std_radrd_;

    R_Lidar_ = MatrixXd(2, 2);
    R_Lidar_ << std_laspx_ * std_laspx_,  0,
                0,                        std_laspy_ * std_laspy_;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    /**
    Complete this function! Make sure you switch between lidar and radar
    measurements.
    */
    if (!is_initialized_){
        // Initalise co-variance matrix
        P_ <<   1,  0,  0,  0,  0,
                0,  1,  0,  0,  0,
                0,  0,  1,  0,  0,
                0,  0,  0,  1,  0,
                0,  0,  0,  0,  1;

        // Identify sensor type and proceed accordingly
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            // Convert radar from polar to cartesian coordinates and initialize state.
            float rho     = meas_package.raw_measurements_(0);
            float phi     = meas_package.raw_measurements_(1);
            float rho_dot = meas_package.raw_measurements_(2);

            float px = rho     * cos(phi);
            float py = rho     * sin(phi);
            float vx = rho_dot * cos(phi);
            float vy = rho_dot * sin(phi);
            float v  = sqrt(vx * vx + vy * vy);

            x_ << px, py, v, 0, 0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

            float px = meas_package.raw_measurements_(0);
            float py = meas_package.raw_measurements_(1);

            if ((fabs(px) < 0.001) && (fabs(py) < 0.001)){
                px = 0.001;
                py = 0.001;
            }
            x_ << px, py, 0, 0, 0;
        }

        weights_(0) = lambda_ / (lambda_ + n_aug_);
        for (int i = 1; i < weights_.size(); i++){
            weights_(i) = 0.5 / (lambda_ + n_aug_);
        }

        time_us_ = meas_package.timestamp_;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    float dt = (meas_package.timestamp_ - time_us_) / 1000000.0;	//dt - expressed in seconds
    time_us_ = meas_package.timestamp_;

    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        // Radar updates
        UpdateRadar(meas_package);

    } else {
        // Laser updates
        UpdateLidar(meas_package);

    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    /**
    Complete this function! Estimate the object's location. Modify the state
    vector, x_. Predict sigma points, the state, and the state covariance matrix.
    */

    // GENERATING SIGMA POINTS
    VectorXd x_aug    = VectorXd(n_aug_);
    MatrixXd P_aug    = MatrixXd(n_aug_, n_aug_);
    MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);

    // create augmented mean state
    x_aug.fill(0.0);
    x_aug.head(n_x_) = x_;

    // create anf fill augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;

    MatrixXd Q = MatrixXd(2, 2);
    Q <<    std_a_ * std_a_,  0,
            0,                std_yawdd_ * std_yawdd_;
    P_aug.bottomRightCorner(2, 2) = Q;

    // create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    // create augmented sigma points
    Xsig_aug.col(0)  = x_aug;
    double lambda_sqrt = sqrt(lambda_+n_aug_);
    VectorXd L_col;
    for (int i = 0; i < n_aug_; i++) {
        L_col = lambda_sqrt * L.col(i);
        Xsig_aug.col(i+1)          = x_aug + L_col;
        Xsig_aug.col(i+1 + n_aug_) = x_aug - L_col;
    }


    // SIGMA POINTS PREDICTIONS
    for (int i = 0; i < n_sigma_; i++) {
        //extract values for better readability
        double p_x      = Xsig_aug(0,i);
        double p_y      = Xsig_aug(1,i);
        double v        = Xsig_aug(2,i);
        double yaw      = Xsig_aug(3,i);
        double yawd     = Xsig_aug(4,i);
        double nu_a     = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > 0.001) {
            px_p = p_x + v/yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v/yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p    = v;
        double yaw_p  = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p  = v_p  + nu_a * delta_t;

        yaw_p  = yaw_p  + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }

    // PREDICT MEAN AND COVARIANCES
    //predicted state mean
    x_ = Xsig_pred_ * weights_;

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = NormaliseAngle(x_diff(3));

        P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    /**
    Complete this function! Use lidar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the lidar NIS.
    */
    int n_z = 2;
    MatrixXd Zsig = Xsig_pred_.block(0, 0, n_z, n_sigma_);

    UpdateAll(meas_package, Zsig, n_z);

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    /**
    Complete this function! Use radar data to update the belief about the object's
    position. Modify the state vector, x_, and covariance, P_.

    You'll also need to calculate the radar NIS.
    */
    int n_z = 3;
    MatrixXd Zsig = MatrixXd(n_z, n_sigma_);

    //transform sigma points into measurement space
    for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v   = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1  = cos(yaw) * v;
        double v2  = sin(yaw) * v;

        // measurement model
        Zsig(0,i)  = sqrt(p_x * p_x + p_y * p_y);                          //r
        Zsig(1,i)  = atan2(p_y,p_x);                                       //phi
        Zsig(2,i)  = (p_x * v1 + p_y * v2 ) / sqrt(p_x * p_x + p_y * p_y); //r_dot
    }

    UpdateAll(meas_package, Zsig, n_z);
}


void UKF::UpdateAll(MeasurementPackage meas_package, MatrixXd Zsig, int n_z){
    //mean predicted measurement
    VectorXd z_pred = VectorXd(n_z);
    z_pred = Zsig * weights_;

    //measurement covariance matrix S
    MatrixXd S = MatrixXd(n_z,n_z);
    S.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        z_diff(1) = NormaliseAngle(z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z, n_z);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
        R = R_Radar_;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER){ // Lidar
        R = R_Lidar_;
    }

    S = S + R;


    // UPDATE STATE
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < n_sigma_; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        z_diff(1) = NormaliseAngle(z_diff(1));

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        x_diff(3) = NormaliseAngle(x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    // Incoming measurment
    VectorXd z = meas_package.raw_measurements_;

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
        // Angle normalization
        z_diff(1) = NormaliseAngle(z_diff(1));
    }

    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // Calculate NIS
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR){ // Radar
        NIS_radar_ = z.transpose() * S.inverse() * z;
        //cout << "NIS RADAR : " << NIS_radar_ << endl;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER){ // Lidar
        NIS_laser_ = z.transpose() * S.inverse() * z;
        //cout << "NIS LASER : " << NIS_radar_ << endl;
    }
}


double UKF::NormaliseAngle(double phi) {
    return atan2(sin(phi), cos(phi));
}
