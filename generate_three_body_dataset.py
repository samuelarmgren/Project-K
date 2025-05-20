import math
import numpy as np
import random

class Body:
    def __init__(self, x, y, vx, vy, mass):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.mass = mass

def gravitational_force(b1, b2, G=1.0, ignore_radius=0.2):
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist_sq = dx**2 + dy**2
    if dist_sq < ignore_radius**2:
        return 0.0, 0.0  # Ignore gravitational force if too close
    dist = math.sqrt(dist_sq + 1e-5)  # avoid exact zero
    force = G * b1.mass * b2.mass / dist_sq
    fx = force * dx / dist
    fy = force * dy / dist
    return fx, fy

def simulate_three_body_system(dt=0.01, time_for_training=10, total_time=100):
    # Random initial conditions
    bodies = [
        Body(random.uniform(-1.1, -0.9), random.uniform(-1.1, -0.9),
             random.uniform(0, 0.05), random.uniform(0, 0.05), 1.0), 
        Body(random.uniform(0.9, 1.1), random.uniform(-1.1, -0.9),
             random.uniform(-0.05, 0), random.uniform(0, 0.05), 1.0),
        Body(random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1),
             random.uniform(-0.05, 0.05), random.uniform(-0.05, 0), 1.0)
    ]
    # Initial half-step velocity update using initial forces
    forces = [(0, 0)] * 3
    for i in range(3):
        for j in range(3):
            if i != j:
                fx, fy = gravitational_force(bodies[i], bodies[j])
                forces[i] = (forces[i][0] + fx, forces[i][1] + fy)
    for i, b in enumerate(bodies):
        fx, fy = forces[i]
        b.vx += 0.5 * fx / b.mass * dt
        b.vy += 0.5 * fy / b.mass * dt

    full_trajectory = []
    total_steps = int(total_time / dt)
    for step in range(total_steps):
        snapshot = []
        for b in bodies:
            snapshot.extend([b.x, b.y, b.vx, b.vy])
        full_trajectory.append(snapshot)

        # Position update using half-step velocities
        for b in bodies:
            b.x += b.vx * dt
            b.y += b.vy * dt

        # Compute forces at new positions
        forces = [(0, 0)] * 3
        for i in range(3):
            for j in range(3):
                if i != j:
                    fx, fy = gravitational_force(bodies[i], bodies[j])
                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)

        # Velocity update to next half-step
        for i, b in enumerate(bodies):
            fx, fy = forces[i]
            b.vx += fx / b.mass * dt
            b.vy += fy / b.mass * dt

    return np.array(full_trajectory)

def generate_dataset(n_samples=1000, dt=0.1, time_for_training=50, total_time=100):
    input_len = int(time_for_training / dt)
    X = []
    y = []
    full_trajs = []

    for i in range(n_samples):
        if i % 10 == 0:
            print(f"{i} samples have been created")
        full_traj = simulate_three_body_system(dt=dt, time_for_training=time_for_training, total_time=total_time)
        X.append(full_traj[:input_len])
        y.append(full_traj[input_len:])
        full_trajs.append(full_traj)


    X = np.array(X)  # shape: (N, input_len, 12)
    y = np.array(y)  # shape: (N, total_len - input_len, 12)
    print(X.shape)
    print(y.shape)
    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("full_traj.npy", np.array(full_trajs))
    print(f"Saved X.npy with shape {X.shape}, y.npy with shape {y.shape}")



if __name__ == "__main__":
    generate_dataset()
