import numpy as np

def generate_eight_trajectory(origin, max_distance, num_loops, loop_duration, delta_time, angle, noise_std=0.01):
    """
    8の字軌道を生成する
    :param origin: 軌道の原点 [x,y]
    :param max_distance: 軌道の原点からの最大距離 [x,y]
    :param num_loops: 軌道のループ数
    :param loop_duration: 1つの軌道を描く時間 [s]
    :param delta_time: 軌道の時間刻み [s]
    :param angle: z軸周りの回転角度 [rad]
    :param noise_std: ホワイトノイズの標準偏差
    :return: 軌道のリスト [(time, x, y)]
    """
    x_origin, y_origin = origin
    max_x, max_y = max_distance
    trajectory = []
    total_time = 0.0

    # 回転行列の計算
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)

    for loop in range(num_loops):
        t = 0.0
        while t < loop_duration:
            # 8の字の軌道を生成
            x = max_x * np.sin(2 * np.pi * t / loop_duration)
            y = max_y * np.sin(4 * np.pi * t / loop_duration)

            # 回転を適用 (originを中心に)
            x_rotated = cos_angle * x - sin_angle * y + x_origin
            y_rotated = sin_angle * x + cos_angle * y + y_origin

            # ホワイトノイズを追加
            x_noisy = x_rotated + np.random.normal(0, noise_std)
            y_noisy = y_rotated + np.random.normal(0, noise_std)

            trajectory.append((total_time, x_noisy, y_noisy))
            t += delta_time
            total_time += delta_time

    return trajectory


def animate_trajectory(trajectory):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.collections import LineCollection
    import numpy as np

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    max_val=np.max(np.array(trajectory)[:,1:])
    min_val=np.min(np.array(trajectory)[:,1:])
    ax.set_xlim([1.2*min_val,1.2*max_val])
    ax.set_ylim([1.2*min_val,1.2*max_val])

    # 軌跡を保持するためのリスト
    x_data, y_data = [], []

    # 初期化
    line_segments = LineCollection([], linewidths=2)
    ax.add_collection(line_segments)

    def init():
        line_segments.set_segments([])
        return line_segments,

    def update(frame):
        x, y = frame[1], frame[2]
        x_data.append(x)
        y_data.append(y)

        # 軌跡のセグメントを作成
        points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # 色を時間に応じて変化させる
        norm = plt.Normalize(0, len(x_data))
        colors = plt.cm.viridis(norm(range(len(x_data))))

        line_segments.set_segments(segments)
        line_segments.set_color(colors)

        return line_segments,

    ani = animation.FuncAnimation(fig, update, frames=trajectory, init_func=init, blit=True, interval=50)
    plt.show()


if __name__=="__main__":
    # 使用例
    origin = (0.5, 0)
    max_distance = (0.3, 0.2)
    angle=np.pi/2
    num_loops = 3
    loop_duration = 20.0  # 1つの軌道を描く時間 [s]
    delta_time = 0.1
    noise_std=0.0005

    trajectory = generate_eight_trajectory(
        origin, max_distance, num_loops, loop_duration, 
        delta_time,angle=angle,noise_std=noise_std
    )
    animate_trajectory(trajectory)