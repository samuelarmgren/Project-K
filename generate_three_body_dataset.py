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

def gravitational_force(b1, b2, G=1.0, softening=0.05):
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist_sq = dx**2 + dy**2 + softening**2
    dist = math.sqrt(dist_sq)
    force = G * b1.mass * b2.mass / dist_sq
    fx = force * dx / dist
    fy = force * dy / dist
    return fx, fy

def simulate_three_body_system(dt=0.005, total_time=4):
    bodies = [
        Body(random.uniform(-1.0, -0.9), random.uniform(-1.0, -0.9), random.uniform(0, 0.05), random.uniform(0, 0.05), 1),
        Body(random.uniform(0.9, 1.0), random.uniform(-1.0, -0.9), random.uniform(-0.05, 0), random.uniform(0, 0.05), 1),
        Body(random.uniform(-0.1, 0.1), random.uniform(0.9, 1.0), random.uniform(-0.05, 0.05), random.uniform(-0.05, 0), 1)
    ]

    total_steps = int(total_time / dt)
    trajectory = []

    # Initial half-step velocity update
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

    for step in range(total_steps):
        snapshot = []
        for i, b in enumerate(bodies):
            # We’ll store acceleration below after force calculation
            snapshot.extend([b.x, b.y, b.vx, b.vy])  # position + velocity

        # Update positions
        for b in bodies:
            b.x += b.vx * dt
            b.y += b.vy * dt

        # Recalculate forces and get acceleration
        forces = [(0, 0)] * 3
        for i in range(3):
            for j in range(3):
                if i != j:
                    fx, fy = gravitational_force(bodies[i], bodies[j])
                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)

        for i, b in enumerate(bodies):
            fx, fy = forces[i]
            ax = fx / b.mass
            ay = fy / b.mass
            #snapshot.extend([ax, ay])
            b.vx += ax * dt
            b.vy += ay * dt

        trajectory.append(snapshot)

    return np.array(trajectory)

def generate_dataset(n_samples=1500, dt=0.005, time_for_training=3, total_time=6):
    input_len = int(time_for_training / dt)
    X = []
    y = []
    full_trajs = []
    shortest = 100000
    for i in range(n_samples):
        
        if i % 10 == 0:
            print(f"{i} samples have been created")
        full_traj = simulate_three_body_system(dt=dt, total_time=total_time)
        if len(full_traj) < shortest:
            shortest = len(full_traj)
            print(shortest)
        
        X.append(full_traj[:input_len])
        y.append(full_traj)
    
        full_trajs.append(full_traj)
    print(shortest)
    
    X = np.array(X) 
    y = np.array(y)  
    print(X.shape)
    print(y.shape)
    np.save("X.npy", X)
    np.save("y.npy", y)
    np.save("full_traj.npy", np.array(full_trajs))
    print(f"Saved X.npy with shape {X.shape}, y.npy with shape {y.shape}")



if __name__ == "__main__":
    generate_dataset()
