import numpy as np
def animate_trajectory(traj, step=5):
    """Faster animation of 3-body trajectory with frame skipping."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')
    ax.set_title('Three-Body Fast Simulation')
    ax.grid(True)
    ax.set_aspect('equal')

    points, = ax.plot([], [], 'bo', markersize=8, animated=True)

    def init():
        points.set_data([], [])
        return points,

    def update(frame):
        idx = frame * step
        if idx >= len(traj):
            idx = len(traj) - 1
        x = [traj[idx, 0], traj[idx, 4], traj[idx, 8]]
        y = [traj[idx, 1], traj[idx, 5], traj[idx, 9]]
        points.set_data(x, y)
        return points,

    total_frames = len(traj) // step
    ani = animation.FuncAnimation(fig, update, frames=total_frames,
                                  init_func=init, blit=True, interval=200, repeat=False)
    plt.show()



traj = np.load("y.npy") 
animate_trajectory(traj[0])
