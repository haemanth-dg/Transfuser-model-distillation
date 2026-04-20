"""
This file holds class implementations for a longitudinal PID controller and a linear regression model as
longitudinal controller.
"""

import numpy as np


class LongitudinalController:
    """
    Base class for longitudinal controller.
    """

    def __init__(self, config):
        """
        Constructor of the longitudinal controller, which saves the configuration object for the hyperparameters.

        Args:
            config (GlobalConfig): Object of the config for hyperparameters.
        """
        self.config = config

    def _safe_fallback(self):
        """
        Fail-safe behavior: no throttle, full brake.
        """
        return 0.0, True

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        """
        Get the throttle and brake values based on the target speed, current speed, and hazard brake condition.
        """
        raise NotImplementedError

    def get_throttle_extrapolation(self, target_speed, current_speed):
        """
        Get the throttle value for the given target speed and current speed.
        """
        raise NotImplementedError

    def save(self):
        """
        Save the current state of the controller.
        """
        pass

    def load(self):
        """
        Load the previously saved state of the controller.
        """
        pass


class LongitudinalPIDController(LongitudinalController):
    """
    PID-based longitudinal controller (used for ablations).
    """

    def __init__(self, config):
        super().__init__(config)

        self.proportional_gain = self.config.longitudinal_pid_proportional_gain
        self.derivative_gain = self.config.longitudinal_pid_derivative_gain
        self.integral_gain = self.config.longitudinal_pid_integral_gain
        self.max_window_length = self.config.longitudinal_pid_max_window_length
        self.speed_error_scaling = self.config.longitudinal_pid_speed_error_scaling
        self.braking_ratio = self.config.longitudinal_pid_braking_ratio
        self.minimum_target_speed = self.config.longitudinal_pid_minimum_target_speed

        self.speed_error_window = []
        self.saved_speed_error_window = []

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        try:
            if hazard_brake or target_speed < 1e-5:
                return 0.0, True

            target_speed = max(self.minimum_target_speed, target_speed)

            current_speed = 3.6 * current_speed
            target_speed = 3.6 * target_speed

            if current_speed / target_speed > self.braking_ratio:
                self.speed_error_window = [0] * self.max_window_length
                return 0.0, True

            speed_error = target_speed - current_speed
            speed_error += speed_error * current_speed * self.speed_error_scaling

            self.speed_error_window.append(speed_error)
            self.speed_error_window = self.speed_error_window[-self.max_window_length:]

            derivative = (
                0 if len(self.speed_error_window) == 1
                else self.speed_error_window[-1] - self.speed_error_window[-2]
            )
            integral = np.mean(self.speed_error_window)

            throttle = (
                self.proportional_gain * speed_error +
                self.derivative_gain * derivative +
                self.integral_gain * integral
            )

            return np.clip(throttle, 0.0, 1.0), False

        except Exception:
            self.speed_error_window = []
            return self._safe_fallback()

    def get_throttle_extrapolation(self, target_speed, current_speed):
        try:
            return self.get_throttle_and_brake(False, target_speed, current_speed)[0]
        except Exception:
            return 0.0

    def save(self):
        self.saved_speed_error_window = self.speed_error_window.copy()

    def load(self):
        self.speed_error_window = self.saved_speed_error_window.copy()


class LongitudinalLinearRegressionController(LongitudinalController):
    """
    Linear regression based longitudinal controller (default).
    """

    def __init__(self, config):
        super().__init__(config)

        self.minimum_target_speed = self.config.longitudinal_linear_regression_minimum_target_speed
        self.params = self.config.longitudinal_linear_regression_params
        self.maximum_acceleration = self.config.longitudinal_linear_regression_maximum_acceleration
        self.maximum_deceleration = self.config.longitudinal_linear_regression_maximum_deceleration

    def get_throttle_and_brake(self, hazard_brake, target_speed, current_speed):
        try:
            if target_speed < 1e-5 or hazard_brake:
                return 0.0, True
            elif target_speed < self.minimum_target_speed:
                target_speed = self.minimum_target_speed

            current_speed *= 3.6
            target_speed *= 3.6
            speed_error = target_speed - current_speed

            if speed_error > self.maximum_acceleration:
                return 1.0, False

            if current_speed / target_speed > self.params[-1]:
                return 0.0, True

            speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
            current_speed /= 100.0

            features = np.array([
                current_speed,
                current_speed ** 2,
                100 * speed_error_cl,
                speed_error_cl ** 2,
                current_speed * speed_error_cl,
                current_speed ** 2 * speed_error_cl
            ])

            throttle = np.clip(features @ self.params[:-1], 0.0, 1.0)
            return throttle, False

        except Exception:
            return self._safe_fallback()

    def get_throttle_extrapolation(self, target_speed, current_speed):
        try:
            current_speed *= 3.6
            target_speed *= 3.6
            speed_error = target_speed - current_speed

            if speed_error > self.maximum_acceleration:
                return 1.0
            elif speed_error < self.maximum_deceleration:
                return 0.0

            if target_speed < 0.1 or current_speed / target_speed > self.params[-1]:
                return 0.0

            speed_error_cl = np.clip(speed_error, 0.0, np.inf) / 100.0
            current_speed /= 100.0

            features = np.array([
                current_speed,
                current_speed ** 2,
                100 * speed_error_cl,
                speed_error_cl ** 2,
                current_speed * speed_error_cl,
                current_speed ** 2 * speed_error_cl
            ])

            return np.clip(features @ self.params[:-1], 0.0, 1.0)

        except Exception:
            return 0.0
