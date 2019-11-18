# -*- coding: utf-8 -*-
# fast_slam_1.py

from collections import namedtuple

import copy
import random
import numpy as np
import matplotlib.pyplot as plt

# 並進に対する共分散のパラメータ
SIGMA_LINEAR_TO_LINEAR = 0.05
# 並進から回転に対する共分散のパラメータ
SIGMA_ANGULAR_TO_LINEAR = 0.05
# 回転から並進に対する共分散のパラメータ
SIGMA_LINEAR_TO_ANGULAR = 0.05
# 回転から回転に対する共分散のパラメータ
SIGMA_ANGULAR_TO_ANGULAR = 0.05
# 並進から到達地点での回転に対する共分散のパラメータ
SIGMA_LINEAR_TO_GAMMA = 0.025
# 回転から到達地点での回転に対する共分散のパラメータ
SIGMA_ANGULAR_TO_GAMMA = 0.025

# パーティクルの個数
NUM_OF_PARTICLES = 30

# ロボットの姿勢(直交座標とヨー角)
RobotPose2D = namedtuple("RobotPose2D", ("x", "y", "theta"))
# ロボットの動作(並進速度と回転速度)
Twist2D = namedtuple("Twist2D", ("linear", "angular"))
# ロボットの計測(距離と方向)
Observation2D = namedtuple("Observation2D", ("r", "phi"))
# 特徴(直交座標)
Feature2D = namedtuple("Feature2D", ("x", "y"))
# パーティクルがもつ地図のパラメータ
FeatureParams2D = namedtuple("FeatureParams2D", ("mu", "sigma"))
# パーティクル(姿勢と地図)
Particle = namedtuple("Particle", ("trajectory", "particle_map"))

# 計測の共分散
SIGMA_OBSERVATION_DISTANCE = 0.1
SIGMA_OBSERVATION_ANGLE = 0.05
Q_t = np.diag([SIGMA_OBSERVATION_DISTANCE, SIGMA_OBSERVATION_ANGLE])

def wrap_theta(theta):
    if theta <= -np.pi:
        theta += 2.0 * np.pi
    elif theta >= np.pi:
        theta -= 2.0 * np.pi
    return theta

# ロボットの動作モデルから, 現在の姿勢を計算
def update_robot_pose_2d(x_prev, u_t, dt):
    # 現在のロボットの並進速度と回転速度
    v, omega = u_t
    # 直前のロボットの姿勢
    x, y, theta = x_prev
    
    # 並進運動のみの場合(omegaが小さい場合)
    if omega < 1e-5:
        x_hat = x + np.cos(theta) * dt
        y_hat = y + np.sin(theta) * dt
        theta_hat = theta
        
        return RobotPose2D(x=x_hat, y=y_hat, theta=theta_hat)
    
    # 現在のロボットの姿勢を計算
    radius = v / omega
    x_hat = x - radius * np.sin(theta) + \
            radius * np.sin(theta + omega * dt)
    y_hat = y + radius * np.cos(theta) - \
            radius * np.cos(theta + omega * dt)
    theta_hat = theta + omega * dt
    theta_hat = wrap_theta(theta_hat)
    
    return RobotPose2D(x=x_hat, y=y_hat, theta=theta_hat)

# ロボットの動作モデルから, 現在の姿勢をサンプリング
def sample_robot_pose_2d(x_prev, u_t, dt):
    # 現在のロボットの並進速度と回転速度
    v, omega = u_t
    
    # 並進速度と回転速度にノイズを加算
    sigma_v = SIGMA_LINEAR_TO_LINEAR * np.power(v, 2.0) + \
              SIGMA_ANGULAR_TO_LINEAR * np.power(omega, 2.0)
    v_hat = v + np.random.normal(scale=sigma_v)
    
    sigma_omega = SIGMA_LINEAR_TO_ANGULAR * np.power(v, 2.0) + \
                  SIGMA_ANGULAR_TO_ANGULAR * np.power(omega, 2.0)
    omega_hat = omega + np.random.normal(scale=sigma_omega)
    
    # 到達地点での回転速度を計算
    sigma_gamma = SIGMA_LINEAR_TO_GAMMA * np.power(v, 2.0) + \
                  SIGMA_ANGULAR_TO_GAMMA * np.power(omega, 2.0)
    gamma_hat = np.random.normal(scale=sigma_gamma)
    
    # 直前のロボットの姿勢
    x, y, theta = x_prev
    
    # 並進運動のみの場合(omega_hatが小さい場合)
    if omega_hat < 1e-5:
        x_hat = x + np.cos(theta) * dt
        y_hat = y + np.sin(theta) * dt
        theta_hat = theta
        
        return RobotPose2D(x=x_hat, y=y_hat, theta=theta_hat)
    
    # 現在のロボットの姿勢にノイズを加算
    radius_hat = v_hat / omega_hat
    x_hat = x - radius_hat * np.sin(theta) + \
                radius_hat * np.sin(theta + omega_hat * dt)
    y_hat = y + radius_hat * np.cos(theta) - \
                radius_hat * np.cos(theta + omega_hat * dt)
    theta_hat = theta + omega_hat * dt + gamma_hat * dt
    theta_hat = wrap_theta(theta_hat)
    
    return RobotPose2D(x=x_hat, y=y_hat, theta=theta_hat)
    
# ロボットの姿勢と特徴から, ロボットの計測を予測
def estimate_observation_2d(x_t, mu_prev):
    # 現在のロボットの姿勢
    x, y, theta = x_t
    
    # 直前の時刻における特徴の予測
    mu_x, mu_y = mu_prev
    
    # ロボットの姿勢と特徴との差分
    dx = mu_x - x
    dy = mu_y - y
    
    # 計測の予測
    dist = np.sqrt(np.power(dx, 2.0) + np.power(dy, 2.0))
    phi = np.arctan2(dy, dx) - theta
    phi = wrap_theta(phi)
    
    return Observation2D(r=dist, phi=phi)

# ロボットの計測の, 特徴による微分を計算
def calculate_jacobian_mu(x_t, mu_prev):
    # 現在のロボットの姿勢
    x, y, theta = x_t
    
    # 直前の時刻における特徴の予測
    mu_x, mu_y = mu_prev
    
    # ロボットの姿勢と特徴との差分
    dx = mu_x - x
    dy = mu_y - y
    
    # ロボットの姿勢と特徴との距離
    dist_sq = np.power(dx, 2.0) + np.power(dy, 2.0)
    dist = np.sqrt(dist_sq)
    
    # ヤコビ行列の各要素の計算
    J = np.zeros(shape=(2, 2))
    
    # 特徴のX座標による, 距離の偏微分
    J[0, 0] = (mu_x - x) / dist
    # 特徴のY座標による, 距離の偏微分
    J[0, 1] = (mu_y - y) / dist
    # 特徴のX座標による, 方向の偏微分
    J[1, 0] = -(mu_y - y) / dist_sq
    # 特徴のY座標による, 方向の偏微分
    J[1, 1] = (mu_x - x) / dist_sq
    
    return J

# ロボットの姿勢と計測から, 特徴を計算
def calculate_feature_2d(x_t, z_t):
    # 現在のロボットの姿勢
    x, y, theta = x_t
    
    # ロボットの計測
    r, phi = z_t
    
    # 特徴の座標を計算
    mu_x = x + r * np.cos(theta + phi)
    mu_y = y + r * np.sin(theta + phi)
    
    return Feature2D(x=mu_x, y=mu_y)

# 対応関係が既知の場合のFastSLAM 1.0アルゴリズム
def fastslam_1_known_correspondence(prev_particles, u_t, z_t, dt):
    # 仮のパーティクルのセットを初期化
    tmp_particles = []
    # パーティクルの重みを初期化
    weights = []
    
    # 各パーティクルに対する処理
    for k in range(NUM_OF_PARTICLES):
        # 直前の時刻におけるパーティクル
        particle = prev_particles[k]
        # 直前の時刻におけるパーティクルがもつ地図
        particle_map = particle.particle_map
        # 直前の時刻におけるパーティクルの軌跡
        trajectory = particle.trajectory
        
        # 新たな姿勢をサンプリング
        pose = sample_robot_pose_2d(trajectory[-1], u_t, dt)
        # 新たな姿勢を軌跡に追加
        trajectory.append(pose)
        
        # パーティクルの重み
        w = 1.0
        
        # 各計測に対する処理
        for i in range(len(z_t)):
            # 計測と対応する特徴とそのインデックス
            j, z_t_i = z_t[i]
            
            if j not in particle_map:
                # パーティクルが特徴を初めて観測した場合
                # 特徴の平均を初期化
                mu = calculate_feature_2d(pose, z_t_i)
                # ヤコビ行列を計算
                H = calculate_jacobian_mu(pose, mu)
                # 特徴の共分散を初期化
                sigma = np.linalg.pinv(H) @ Q_t @ np.linalg.pinv(H).T
                # 重みを初期化
                w *= 1.0 / NUM_OF_PARTICLES
                # w *= 1.0
                # 特徴の平均と共分散を更新
                particle_map[j] = FeatureParams2D(mu=mu, sigma=sigma)
            else:
                # パーティクルが特徴を既に観測している場合
                mu = particle_map[j].mu
                sigma = particle_map[j].sigma
                
                # 計測の予測を計算
                z_hat = estimate_observation_2d(pose, mu)
                # ヤコビ行列を計算
                H = calculate_jacobian_mu(pose, mu)
                # 計測の共分散を計算
                Q = H @ sigma @ H.T + Q_t
                # カルマンゲインを計算
                K = sigma @ H.T @ np.linalg.pinv(Q)
                # イノベーションベクトルを計算
                d = np.array([z_t_i.r, z_t_i.phi]) - \
                    np.array([z_hat.r, z_hat.phi])
                d[1] = wrap_theta(d[1])
                # 特徴の平均を計算
                mu = mu + K @ d
                # 特徴の共分散を計算
                sigma = sigma - sigma @ K @ H
                # 重みを計算
                w *= np.power(np.linalg.det(Q), 0.5) * \
                     np.exp(-0.5 * (d.T @ np.linalg.pinv(Q) @ d))
                # w *= np.exp(-0.01 * (d.T @ np.linalg.pinv(Q) @ d))
                # 特徴の平均と共分散を更新
                particle_map[j] = FeatureParams2D(mu=mu, sigma=sigma)
        
        # 新たなパーティクルを作成して追加
        new_particle = Particle(trajectory=trajectory,
                                particle_map=particle_map)
        tmp_particles.append(new_particle)
        weights.append(w)
    
    # 重みの正規化
    sum_weight = np.sum(weights)
    
    if sum_weight == 0.0:
        print("Warning: weights are reset, since sum of weights is 0")
        weights = [1.0 / NUM_OF_PARTICLES for x in range(NUM_OF_PARTICLES)]
    else:
        weights = weights / sum_weight
    
    # リサンプリング処理
    new_particle_indices = np.random.choice(
        NUM_OF_PARTICLES, size=NUM_OF_PARTICLES, p=weights)
    particles = [copy.deepcopy(tmp_particles[i]) for i in new_particle_indices]
    
    return particles, weights

# ロボットの座標が特徴と十分に近いかどうか判定
def is_close_enough(pose, feature):
    # ロボットの姿勢
    x, y, theta = pose
    # 特徴の座標
    mx, my = feature
    
    dist_sq = np.power(mx - x, 2.0) + np.power(my - y, 2.0)
    dist = np.sqrt(dist_sq)
    
    return dist < 5.0

# ロボットの観測にガウス雑音を加算
def add_gaussian_noise(z_t):
    # ロボットの観測
    r, phi = z_t
    
    # 適当なガウス雑音を加算
    r_hat = r + np.random.normal(scale=0.1)
    phi_hat = phi + np.random.normal(scale=0.05)
    phi_hat = wrap_theta(phi_hat)
    
    return Observation2D(r=r_hat, phi=phi_hat)

# 特徴の真の座標を生成
def generate_simulated_features_old():
    features = []
    
    # 内側の壁
    feature_x0 = np.linspace(-3.0, 3.0, 11)
    feature_y0 = np.repeat(2.0, 11)
    feature_y1 = np.repeat(8.0, 11)
    features.extend(np.stack([feature_x0, feature_y0], axis=1))
    features.extend(np.stack([feature_x0, feature_y1], axis=1))
    
    feature_x0 = np.repeat(-3.0, 11)[1:-1]
    feature_x1 = np.repeat(3.0, 11)[1:-1]
    feature_y0 = np.linspace(2.0, 8.0, 11)[1:-1]
    features.extend(np.stack([feature_x0, feature_y0], axis=1))
    features.extend(np.stack([feature_x1, feature_y0], axis=1))
    
    # 外側の壁
    features_x0 = np.linspace(-6.0, 6.0, 21)
    features_y0 = np.repeat(-1.0, 21)
    features_y1 = np.repeat(11.0, 21)
    features.extend(np.stack([features_x0, features_y0], axis=1))
    features.extend(np.stack([features_x0, features_y1], axis=1))
    
    features_x0 = np.repeat(-6.0, 21)[1:-1]
    features_x1 = np.repeat(6.0, 21)[1:-1]
    features_y0 = np.linspace(-1.0, 11.0, 21)[1:-1]
    features.extend(np.stack([features_x0, features_y0], axis=1))
    features.extend(np.stack([features_x1, features_y0], axis=1))
    
    # Feature2Dのリストに変換
    features = [Feature2D(x=f[0], y=f[1]) for f in features]
    
    return features

# 特徴の真の座標を生成
def generate_simulated_features():
    features = []

    for i in range(20):
        r = 3.0 + np.random.normal(scale=0.3)
        phi = 2.0 * np.pi * i / 20 + np.random.normal(scale=0.1)

        feature_x = 0.0 + r * np.cos(phi)
        feature_y = 5.0 + r * np.sin(phi)

        features.append([feature_x, feature_y])
    
    for i in range(20):
        r = 6.0 + np.random.normal(scale=0.3)
        phi = 2.0 * np.pi * i / 20 + np.random.normal(scale=0.1)
        
        feature_x = 0.0 + r * np.cos(phi)
        feature_y = 5.0 + r * np.sin(phi)

        features.append([feature_x, feature_y])
    
    features = [Feature2D(x=f[0], y=f[1]) for f in features]

    return features

# ロボットの動作と計測を生成
def generate_simulated_twist_and_observation():
    # 特徴の真の座標を生成
    features = generate_simulated_features()
    
    # ロボットの真の位置, 制御, 観測を生成
    ground_truth = []
    twists = []
    observations = []
    deltas = []
    
    pose = RobotPose2D(x=0.0, y=0.0, theta=0.0)
    twist = Twist2D(linear=0.5, angular=0.1)
    
    ground_truth.append(pose)
    
    for i in range(256):
        dt = max(0.2 + np.random.normal(scale=0.05), 0.15)
        deltas.append(dt)
        
        # ロボットの真の位置を追加
        pose = update_robot_pose_2d(pose, twist, dt)
        ground_truth.append(pose)
        
        # ロボットの動作に適当なノイズを加算して追加
        v_hat = twist.linear + np.random.normal(scale=0.1)
        omega_hat = twist.angular + np.random.normal(scale=0.05)
        twist_hat = Twist2D(linear=v_hat, angular=omega_hat)
        twists.append(twist_hat)
        
        # ロボットが観測可能な特徴とそのインデックス
        possible_features = list(filter(
            lambda f: is_close_enough(pose, f[1]),
            enumerate(features)))
        # 特徴からロボットの観測を計算
        possible_observations = list(map(
            lambda f: (f[0], estimate_observation_2d(pose, f[1])),
            possible_features))
        
        observations.append(possible_observations)
        
    return ground_truth, twists, observations, deltas, features

def main():
    # 入力データを作成
    ground_truth, twists, observations, deltas, features = \
        generate_simulated_twist_and_observation()
    
    # パーティクルを初期化
    particles = []

    for i in range(NUM_OF_PARTICLES):
        init_x = ground_truth[0].x + np.random.normal(scale=0.02)
        init_y = ground_truth[0].y + np.random.normal(scale=0.02)
        init_theta = ground_truth[0].theta + np.random.normal(scale=0.02)
        init_pose = RobotPose2D(x=init_x, y=init_y, theta=init_theta)
        particles.append(Particle(trajectory=[init_pose], particle_map={}))
    
    weights = []

    # オドメトリのみを用いた軌跡
    odom_pose = ground_truth[0]
    odom_trajectory = [odom_pose]
    
    for i in range(len(twists)):
        odom_pose = update_robot_pose_2d(odom_pose, twists[i], deltas[i])
        odom_trajectory.append(odom_pose)
    
    # FastSLAM 1.0アルゴリズムを実行
    for i in range(len(twists)):
        print("Iteration: {0}".format(i))
        particles, weights = fastslam_1_known_correspondence(
            particles, twists[i], observations[i], deltas[i])
    
    # 特徴
    feature_xs = [f.x for f in features]
    feature_ys = [f.y for f in features]
    plt.scatter(feature_xs, feature_ys,
                s=20, c="green", label="Features")
    
    # 正しい軌跡
    ground_truth_xs = [pose.x for pose in ground_truth]
    ground_truth_ys = [pose.y for pose in ground_truth]
    plt.plot(ground_truth_xs, ground_truth_ys,
             linewidth=3, c="blue", label="Ground truth")
    
    # オドメトリのみを用いた場合の軌跡
    odom_trajectory_xs = [t.x for t in odom_trajectory]
    odom_trajectory_ys = [t.y for t in odom_trajectory]
    plt.plot(odom_trajectory_xs, odom_trajectory_ys,
             linewidth=3, c="black", label="Odometry only")
    
    # 最良のパーティクルの軌跡
    best_particle_idx = np.argmax(weights)
    best_particle_trajectory = particles[best_particle_idx].trajectory
    trajectory_xs = [t.x for t in best_particle_trajectory]
    trajectory_ys = [t.y for t in best_particle_trajectory]
    plt.plot(trajectory_xs, trajectory_ys,
             linewidth=3, c="red", label="Best particle trajectory")
    
    plt.xlabel("x")
    plt.ylabel("y")
    
    plt.axis("scaled")
    plt.legend(loc="center",
               bbox_to_anchor=(0.5, -0.15),
               borderaxespad=0)
    plt.subplots_adjust(bottom=0.2)
    plt.title("FastSLAM 1.0")
    plt.show()

if __name__ == "__main__":
    main()
    