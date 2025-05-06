
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

def gravitational_force(b1, b2, G=1.0):
    dx = b2.x - b1.x
    dy = b2.y - b1.y
    dist_sq = dx**2 + dy**2 + 1e-5  # avoid division by zero
    dist = math.sqrt(dist_sq)
    force = G * b1.mass * b2.mass / dist_sq
    fx = force * dx / dist
    fy = force * dy / dist
    return fx, fy

def simulate_three_body_system(energy, dt=0.01, steps=100):
    # Random initial conditions
    if energy == "small":
        bodies = [
            Body(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1),
                random.uniform(-0.05, 0.05), random.uniform(-0.05, 0.05), 1.0)
            for _ in range(3)
        ]
    elif energy == "mid":
        bodies = [
            Body(random.uniform(-1, 1), random.uniform(-1, 1),
                random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), 1.0)
            for _ in range(3)
        ]
    elif energy == "large":
        bodies = [
            Body(random.uniform(-10, 10), random.uniform(-10, 10),
                random.uniform(-5, 5), random.uniform(-5, 5), 1.0)
            for _ in range(3)
        ]
    trajectory = []

    for _ in range(steps):
        snapshot = []
        for b in bodies:
            snapshot.extend([b.x, b.y, b.vx, b.vy])
        trajectory.append(snapshot)

        # Compute forces
        forces = [(0, 0)] * 3
        for i in range(3):
            for j in range(3):
                if i != j:
                    fx, fy = gravitational_force(bodies[i], bodies[j])
                    forces[i] = (forces[i][0] + fx, forces[i][1] + fy)

        # Update velocity and position
        for i, b in enumerate(bodies):
            fx, fy = forces[i]
            b.vx += fx / b.mass * dt
            b.vy += fy / b.mass * dt
            b.x += b.vx * dt
            b.y += b.vy * dt

    return np.array(trajectory), bodies

def label_trajectory(bodies):
    # Final distances between bodies
    def dist(b1, b2):
        return math.sqrt((b1.x - b2.x)**2 + (b1.y - b2.y)**2)

    d12 = dist(bodies[0], bodies[1])
    d13 = dist(bodies[0], bodies[2])
    d23 = dist(bodies[1], bodies[2])

    max_dist = max(d12, d13, d23)
    min_dist = min(d12, d13, d23)

    if max_dist > 5.0:
        return 2  # Divergent
    elif min_dist < 0.2:
        return 0  # Convergent
    else:
        return 1  # Stable

def generate_dataset(n_samples=10000):
    X = []
    y = []
    for i in range(n_samples):
        number = random.randint(1, 3)
        energy_choice = {1: "small", 2: "mid", 3: "large"}
        traj, final_bodies = simulate_three_body_system(energy=energy_choice[number])
        label = label_trajectory(final_bodies)
        X.append(traj)
        y.append(label)
    X = np.array(X)  # shape: (N, 10, 12)
    y = np.array(y)
    np.save("X.npy", X)
    np.save("y.npy", y)
    print(f"Saved X.npy with shape {X.shape}, y.npy with shape {y.shape}")

if __name__ == "__main__":
    generate_dataset()
