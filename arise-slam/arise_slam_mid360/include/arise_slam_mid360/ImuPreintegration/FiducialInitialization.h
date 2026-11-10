#pragma once

#include <gtsam/geometry/Pose3.h>
#include <Eigen/Cholesky>
#include <stdexcept>

namespace arise_slam
{

struct FiducialInitialization
{
  gtsam::Pose3 virtualLidarPose;
  gtsam::Pose3 fiducial_T_world;
  gtsam::Matrix66 imuCovariance;
};

// ROS covariance uses world translation followed by world rotation increments.
// ARISE gravity-aligns raw points before mapping; GTSAM estimates the IMU pose.
inline FiducialInitialization makeFiducialInitialization(
  const gtsam::Pose3 & rawPose, const gtsam::Matrix66 & rosCovariance,
  const gtsam::Rot3 & rawToVirtualRotation, const gtsam::Pose3 & lidar2Imu,
  const gtsam::Pose3 & nativeLidarPose)
{
  if (
    !rosCovariance.allFinite() || !rosCovariance.isApprox(rosCovariance.transpose(), 1e-8) ||
    Eigen::LLT<gtsam::Matrix66>(rosCovariance).info() != Eigen::Success) {
    throw std::invalid_argument("Fiducial covariance must be symmetric positive definite");
  }
  const gtsam::Pose3 rawToVirtual(rawToVirtualRotation, gtsam::Point3(0, 0, 0));
  const auto virtualPose = rawPose.compose(rawToVirtual);
  const auto imuPose = virtualPose.compose(lidar2Imu);
  gtsam::Matrix66 jacobian;
  const double step = 1e-6;
  for (int axis = 0; axis < 6; ++axis) {
    gtsam::Vector6 errors[2];
    for (int sign = 0; sign < 2; ++sign) {
      const double delta = (sign == 0 ? step : -step);
      auto position = rawPose.translation();
      auto rotation = rawPose.rotation();
      if (axis < 3) {
        position(axis) += delta;
      } else {
        gtsam::Vector3 increment = gtsam::Vector3::Zero();
        increment(axis - 3) = delta;
        rotation = gtsam::Rot3::Expmap(increment).compose(rotation);
      }
      const auto perturbed =
        gtsam::Pose3(rotation, position).compose(rawToVirtual).compose(lidar2Imu);
      errors[sign] = gtsam::Pose3::Logmap(imuPose.between(perturbed));
    }
    jacobian.col(axis) = (errors[0] - errors[1]) / (2.0 * step);
  }
  const gtsam::Matrix66 covariance = jacobian * rosCovariance * jacobian.transpose();
  return {
    virtualPose, virtualPose.compose(nativeLidarPose.inverse()),
    (covariance + covariance.transpose()) * 0.5};
}

}  // namespace arise_slam
