#ifndef __SIMPLE_POSITION_CONTROLLER
#define __SIMPLE_POSITION_CONTROLLER
#include <Eigen/Eigen>
#include "HoverThrustEkf.hpp"
#include "Derivative.hpp"

enum control_mode
{
    CTRL_ALL,
    CTRL_POS_ONLY,
    CTRL_VEL_ONLY
};

double fromQuaternion2yaw(Eigen::Quaterniond q)
{
    double yaw = atan2(2 * (q.x() * q.y() + q.w() * q.z()), q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z());
    return yaw;
}
class SimplePositionController
{
private:
    HoverThrustEkf *hoverThrustEkf;
    Derivate velDerivateZ_;

    Eigen::Vector3d _pos_world;
    Eigen::Vector3d _vel_world;
    Eigen::Quaterniond _q_world;
    Eigen::Vector3d _angular_vel_world;

    double _hover_thrust{0.5}; // 不能是0

    double _thrust_sp{0.0};
    Eigen::Quaterniond q_sp;

    Eigen::Vector3d _Kp, _Kv, _Ki;
    Eigen::Vector3d _vel_int{Eigen::Vector3d::Zero()}; // velocity-error integrator (PX4 MPC_*_VEL_I_ACC)
    Eigen::Vector3d _vel_sp_last{Eigen::Vector3d::Zero()}; // previous velocity setpoint, used to gate HTE updates
    double _vel_int_lim{2.0};                          // anti-windup clamp on integral accel [m/s^2]
    double _dt{0.0};                                   // last control dt, cached from set_status

    control_mode _mode = control_mode::CTRL_ALL;


public:
    void set_pid_params(Eigen::Vector3d pos_gains, Eigen::Vector3d vel_gains)
    {
        _Kp = pos_gains;
        _Kv = vel_gains;
    };
    double get_hover_thrust()
    {
        return _hover_thrust;
    };
    // zero the velocity integrator on episode reset (avoids cross-episode windup)
    void reset()
    {
        _vel_int.setZero();
        _vel_sp_last.setZero();
        _thrust_sp = 0.0;
    };
    void set_status(Eigen::Vector3d pos, Eigen::Vector3d vel, Eigen::Vector3d angular_velocity, Eigen::Vector4d q, double dt)
    {
        _vel_world = vel;
        _pos_world = pos;
        _q_world.w() = q(0);
        _q_world.x() = q(1);
        _q_world.y() = q(2);
        _q_world.z() = q(3);
        _angular_vel_world = angular_velocity;
        _dt = dt; // cache control dt for the velocity integrator in update()

        // _hover_thrust = hover_thrust_estimator(0.005, _vel_world(2), thrust_z(2));
        Eigen::Vector3d thrust_z = _q_world.toRotationMatrix() * Eigen::Vector3d(0, 0, _thrust_sp);

        hoverThrustEkf->predict(dt); // dt
        double acc_z = velDerivateZ_.update(_vel_world(2), dt);

        // The hover-thrust EKF is only valid near hover. During random policy commands,
        // aggressive tilt, or large velocity transients, vertical acceleration is command-
        // induced and must not be interpreted as hover-thrust error.
        const double tilt_cos = (_q_world.toRotationMatrix() * Eigen::Vector3d::UnitZ())(2);
        const bool near_level = tilt_cos > std::cos(0.35); // tilt < ~20 deg
        const bool near_velocity = _vel_world.head<2>().norm() < 1.0 && std::abs(_vel_world(2)) < 0.5;
        const bool near_command = _vel_sp_last.head<2>().norm() < 0.5 && std::abs(_vel_sp_last(2)) < 0.25;
        const bool valid_thrust = _thrust_sp > 0.05 && _thrust_sp < 0.95;

        if (near_level && near_velocity && near_command && valid_thrust) {
            hoverThrustEkf->fuseAccZ(acc_z, thrust_z(2));
        }
        // PX4-faithful: use the online HoverThrustEKF estimate as the thrust feedforward.
        // Adapts per-env (handles mass domain randomization); seeded near the 0.30 kg
        // nominal in the constructor so it does not sag while converging.
        _hover_thrust =  hoverThrustEkf->getHoverThrust();
        // _hover_thrust = 0.631; // Starling 2 (0.30 kg): per-motor hover throttle s.t. 4*kT*omega(u)^2 = m*g

    };
    Eigen::VectorXd update(const Eigen::Vector3d &pos_sp, const Eigen::Vector3d &vel_sp, const Eigen::Vector3d &acc_sp, const double yaw_sp);
    void set_control_mode(control_mode mode);
    SimplePositionController(/* args */);
    ~SimplePositionController();
};

SimplePositionController::SimplePositionController(/* args */)
{
    _Kp << 1.5, 1.5, 1.5; // unused in CTRL_VEL_ONLY (velocity command mode)
    _Kv << 3.0, 3.0, 8.0; // MPC_XY_VEL_P_ACC=3.0 (x,y), MPC_Z_VEL_P_ACC=8.0 (z)
    _Ki << 0.1, 0.1, 2.0; // MPC_XY_VEL_I_ACC=0.1 (x,y), MPC_Z_VEL_I_ACC=2.0 (z): from real Starling2 px4-param 
    // With THR_MDL_FAC enabled in the mixer, the controller operates in PX4 thrust-setpoint space.
    // Seed at the SIM's measured hover setpoint, not the real MPC_THR_HOVER=0.34: the SysID
    // throttle_to_thrust curve (at 7.97 V) makes weight at PWM u~=0.64, which through
    // THR_MDL_FAC=0.9 -> setpoint ~=0.43. Seeding at 0.34 left a ~0.09 thrust deficit, so the
    // drone sank on step 1 and the HoverThrustEKF ratcheted upward (the |vz|<0.5 gate let it
    // raise during the slow sink but blocked the corrective fusion during recovery).
    // HTE_HT_ERR_INIT=0.1, HTE_HT_NOISE=0.0036 from the real px4-param dump.
    hoverThrustEkf = new HoverThrustEkf(0.43f, 0.1f, 0.0036f);
}

SimplePositionController::~SimplePositionController()
{
}

void SimplePositionController::set_control_mode(control_mode mode)
{
    _mode = mode;
}
Eigen::VectorXd SimplePositionController::update(const Eigen::Vector3d &pos_sp, const Eigen::Vector3d &vel_sp, const Eigen::Vector3d &acc_sp, const double yaw_sp)
{
    // std::cout << "pos_sp " <<pos_sp<<std::endl;
    // std::cout << "vel_sp " <<vel_sp<<std::endl;
    // std::cout << "acc_sp " <<acc_sp<<std::endl;
    // std::cout << "yaw_sp " <<yaw_sp<<std::endl;

    // compute disired acceleration
    _vel_sp_last = vel_sp;
    Eigen::Vector3d des_acc(0.0, 0.0, 0.0);
    Eigen::Vector3d Kp, Kv;


    if(_mode == CTRL_POS_ONLY)
    {
        des_acc = acc_sp + _Kp.asDiagonal() * (pos_sp - _pos_world);
    }
    else if(_mode == CTRL_VEL_ONLY)
    {
        // PX4 velocity PI: proportional + integral (with anti-windup) on velocity error.
        Eigen::Vector3d vel_err = vel_sp - _vel_world;
        _vel_int += _Ki.asDiagonal() * vel_err * _dt;
        for (int i = 0; i < 3; i++) {
            _vel_int(i) = MyMath::constrain(_vel_int(i), -_vel_int_lim, _vel_int_lim);
        }
        des_acc = acc_sp + _Kv.asDiagonal() * vel_err + _vel_int;
    }
    else
    {
        Eigen::Vector3d vel_err = vel_sp - _vel_world;
        _vel_int += _Ki.asDiagonal() * vel_err * _dt;
        for (int i = 0; i < 3; i++) {
            _vel_int(i) = MyMath::constrain(_vel_int(i), -_vel_int_lim, _vel_int_lim);
        }
        des_acc = acc_sp + _Kv.asDiagonal() * vel_err + _vel_int + _Kp.asDiagonal() * (pos_sp - _pos_world);
    }

	for (int i = 0; i < 3; i++) {
		des_acc(i) = MyMath::constrain(des_acc(i), double(-3.f),double(3.f)); // MPC_ACC_HOR / MPC_ACC_UP_MAX / MPC_ACC_DOWN_MAX = 3.0
	}
    // des_acc += Eigen::Vector3d(0, 0, 0);
    // std::cout << "des_acc " <<des_acc<<std::endl;


    _thrust_sp = des_acc(2) * (_hover_thrust / CONSTANTS_ONE_G) + _hover_thrust;

    double roll, pitch, yaw, yaw_imu;
    double yaw_odom = fromQuaternion2yaw(_q_world);
    double sin = std::sin(yaw_odom);
    double cos = std::cos(yaw_odom);
    roll = (des_acc(0) * sin - des_acc(1) * cos) / CONSTANTS_ONE_G;
    pitch = (des_acc(0) * cos + des_acc(1) * sin) / CONSTANTS_ONE_G;

    q_sp = Eigen::AngleAxisd(yaw_sp, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX());
    
    Eigen::VectorXd atti_thrust_sp(5);
    atti_thrust_sp(0) = q_sp.w();
    atti_thrust_sp(1) = q_sp.x();
    atti_thrust_sp(2) = q_sp.y();
    atti_thrust_sp(3) = q_sp.z();
    atti_thrust_sp(4) = _thrust_sp;

    return atti_thrust_sp;
}
#endif