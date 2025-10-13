"""
Particle class for DEM simulation
"""
import numpy as np


class Particle:
    """
    A particle class for particles with position, velocity, radius and mass.
    """
    
    def __init__(self, x, y, vx=0, vy=0, radius=10, density=1.0):
        """
        Initializing a particle.
        
        Args:
            x, y: Initial position
            vx, vy: Initial velocity
            radius: Particle radius
            density: Material density
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.radius = radius
        self.density = density
        self.mass = np.pi * radius**2 * density
        
        # Collision tracking (Sprint 2)
        self.collision_energy = 0.0
        self.in_collision = False

    def update(self, dt):
        """
        Particle postion update depends on its velocity.
        
        Args:
            dt: Time step
        """
        self.position += self.velocity * dt
        
    def apply_force(self, force, dt):
        """
        Apply a force to the particle for time dt.
        
        Args:
            force: Force vector
            dt: Time step
        """
        acceleration = force / self.mass
        self.velocity += acceleration * dt
        
    def __repr__(self):
        return f"Particle(pos={self.position}, vel={self.velocity}, r={self.radius}, m={self.mass:.2f})"
