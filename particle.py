"""
Particle class for DEM simulation
"""
import numpy as np

from materials import get_material_profile


class Particle:
    """
    Particle with position, velocity, radius, mass, and material metadata.
    """

    def __init__(self, x, y, vx=0, vy=0, radius=10, density=None, material="rock"):
        """
        Initialize a particle.

        Args:
            x, y: Initial position
            vx, vy: Initial velocity
            radius: Particle radius
            density: Optional override for material density
            material: Material name defined in materials.MATERIAL_LIBRARY
        """
        self.position = np.array([x, y], dtype=float)
        self.velocity = np.array([vx, vy], dtype=float)
        self.radius = radius

        self.material_profile = get_material_profile(material)
        self.material = self.material_profile.name
        self.breakage_factor = self.material_profile.breakage_factor
        self.material_color = self.material_profile.color

        resolved_density = density if density is not None else self.material_profile.density
        self.density = resolved_density
        self.mass = np.pi * radius**2 * resolved_density

        # Collision tracking (Sprint 2)
        self.collision_energy = 0.0
        self.in_collision = False

    def update(self, dt):
        """
        Particle position update depends on its velocity.

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
        return (
            f"Particle(material={self.material}, pos={self.position}, "
            f"vel={self.velocity}, r={self.radius}, m={self.mass:.2f})"
        )
