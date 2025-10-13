"""Main simulation engine for DEM
"""
import numpy as np
from particle import Particle


class DEMSimulation:
    """
    DEM simulation with particles and gravity.
    """
    
    def __init__(self, width=800, height=600, gravity=9.81):
        """
        Initializing the simulation.
        
        Args:
            width: Simualtion area width
            height: Simualtion area width
            gravity: Gravitational acceleration
        """
        self.width = width
        self.height = height
        self.gravity = gravity * 100
        self.particles = []
        self.time = 0.0
        
        # Coefficient of restitution (COR) # TODO: Check https://www.sciencedirect.com/science/article/abs/pii/S0921883121000698
        self.restitution = 0.8
        
        # Sprint 2: Collision tracking
        self.total_collisions = 0
        self.collision_energies = []

        # Sprint 3: Fragmentation parameters
        self.energy_threshold = 500.0  # Energy required to break a particle
        self.min_fragment_radius = 5.0  # Minimum size before particles can't break
        self.num_fragments = 3  # Number of fragments per breakage # TODO: Find optimal fragments
        self.total_fragments_created = 0
        self.particles_to_add = []  # Buffer for new fragments
        self.particles_to_remove = []  # Buffer for broken particles
        self.max_particles = 150  # Safety cap
        self.initial_total_energy = 0.0
        self.energy_lost_to_fragmentation = 0.0

    def add_particle(self, particle):
        """Add particle"""
        if len(self.particles) < self.max_particles:
            self.particles.append(particle)
            return True
        else:
            print(f"Particle limit reached ({self.max_particles})")
            return False

    def apply_gravity(self, dt):
        """Apply gravitational force"""
        gravity_force = np.array([0, self.gravity])
        for particle in self.particles:
            particle.apply_force(gravity_force * particle.mass, dt)


    def detect_collision(self, p1, p2): # can be static tho
        """
        Check if two particles are colliding.
        Args:
            p1, p2: Particle objects

        Returns:
            bool: True if particles overlap
        """
        distance = np.linalg.norm(p1.position - p2.position)
        return distance < (p1.radius + p2.radius)

    def resolve_collision(self, p1, p2):
        """
        Resolve collision between two particles using elastic collision physics.
        Conserves momentum and calculates impact energy.

        Args:
            p1, p2: Particle objects that are colliding
        """
        # Vector from p1 to p2
        delta = p2.position - p1.position
        distance = np.linalg.norm(delta)

        if distance == 0:
            distance = p1.radius + p2.radius
            delta = np.array([1.0, 0.0])

        # Normalized collision normal
        normal = delta / distance

        # Relative velocity
        relative_velocity = p1.velocity - p2.velocity

        # Velocity along collision normal
        velocity_along_normal = np.dot(relative_velocity, normal)

        # Don't resolve if particles are moving apart
        if velocity_along_normal < 0:
            return

        # Calculate impact energy
        reduced_mass = (p1.mass * p2.mass) / (p1.mass + p2.mass)
        impact_energy = 0.5 * reduced_mass * np.linalg.norm(relative_velocity)**2

        # Store energy in both particles
        p1.collision_energy = impact_energy
        p2.collision_energy = impact_energy
        p1.in_collision = True
        p2.in_collision = True

        # Track collision statistics
        self.total_collisions += 1
        self.collision_energies.append(impact_energy)

        # Check for fragmentation
        fragmented = False
        if impact_energy > self.energy_threshold:
            # Mark particles for fragmentation if large enough
            if p1.radius >= self.min_fragment_radius:
                self.mark_for_fragmentation(p1, impact_energy)
                fragmented = True
            if p2.radius >= self.min_fragment_radius:
                self.mark_for_fragmentation(p2, impact_energy)
                fragmented = True

        if impact_energy > 100 or fragmented:  # high energy collisions and fragmentations
            frag_text = " [FRAGMENTATION]" if fragmented else ""
            print(f"Collision #{self.total_collisions}: Energy = {impact_energy:.2f} J{frag_text}")

        # Only apply collision response if neither particle fragmenting
        if not fragmented:
            # Elastic collision
            restitution_factor = 1.0 + self.restitution
            impulse = restitution_factor * velocity_along_normal / (1/p1.mass + 1/p2.mass)

            # Apply impulse to both particles
            p1.velocity -= (impulse / p1.mass) * normal
            p2.velocity += (impulse / p2.mass) * normal

        # Separate overlapping particles
        overlap = (p1.radius + p2.radius) - distance
        if overlap > 0:
            # Move particles apart proportional to their masses
            separation = normal * overlap * 0.5
            p1.position -= separation
            p2.position += separation

    def mark_for_fragmentation(self, particle, impact_energy):
        """
        Mark a particle for fragmentation.
        Args:
            particle: Particle to fragment
            impact_energy: Energy of the collision that caused fragmentation
        """
        future_particle_count = len(self.particles) + self.num_fragments - 1
        if future_particle_count > self.max_particles:
            print(f"  -> Fragmentation skipped (would exceed particle limit)")
            return
        if particle not in self.particles_to_remove:
            self.particles_to_remove.append(particle)
            particle.should_fragment = True
            particle.fragment_energy = impact_energy

    def fragment_particle(self, particle):
        """
        Break a particle into smaller fragments.
        Args:
            particle: The particle to fragment

        Returns:
            list: New fragment particles
        """
        fragments = []

        # Calculate fragment properties
        # Set radius, distribute volume among fragments by area
        original_area = np.pi * particle.radius**2
        fragment_area = original_area / self.num_fragments
        fragment_radius = np.sqrt(fragment_area / np.pi)

        # Ensure fragments not too small
        fragment_radius = max(fragment_radius, self.min_fragment_radius)

        # Distribute mass among fragments
        fragment_mass = particle.mass / self.num_fragments
        fragment_density = fragment_mass / (np.pi * fragment_radius**2)

        # Create fragments around the original particle position
        angle_step = 2 * np.pi / self.num_fragments

        # Energy loss during fragmentation due to breaking bonds
        fragmentation_efficiency = 0.7  #kinetic energy preservation

        # Calculate how much energy to add from the collision
        collision_energy = getattr(particle, 'fragment_energy', 0)
        energy_per_fragment = (collision_energy * (1 - fragmentation_efficiency)) / self.num_fragments  #

        for i in range(self.num_fragments):
            angle = i * angle_step + np.random.uniform(-0.3, 0.3)  # randomizer

            # Offset fragments from center
            offset_distance = particle.radius * 0.6
            offset_x = offset_distance * np.cos(angle)
            offset_y = offset_distance * np.sin(angle)

            # Fragment position
            frag_x = particle.position[0] + offset_x
            frag_y = particle.position[1] + offset_y

            # Inherit parent velocity + radial component
            explosion_speed = np.sqrt(2 * energy_per_fragment / fragment_mass) if fragment_mass > 0 else 30
            explosion_speed = min(explosion_speed, 80)  # Cap maximum explosion speed

            frag_vx = particle.velocity[0] * fragmentation_efficiency + explosion_speed * np.cos(angle)
            frag_vy = particle.velocity[1] * fragmentation_efficiency + explosion_speed * np.sin(angle)

            # Create fragment
            fragment = Particle(
                x=frag_x,
                y=frag_y,
                vx=frag_vx,
                vy=frag_vy,
                radius=fragment_radius,
                density=fragment_density
            )

            fragments.append(fragment)

        self.total_fragments_created += self.num_fragments
        print(f"  -> Fragmentation occured. Particle (r={particle.radius:.1f}) broke into {self.num_fragments} fragments (r={fragment_radius:.1f} each)")

        return fragments

    def process_fragmentations(self):
        """
        Process all particles marked for fragmentation.
        Remove broken particles and add their fragments.
        """
        # Remove broken particles and create fragments
        for particle in self.particles_to_remove:
            if particle in self.particles:
                fragments = self.fragment_particle(particle)
                self.particles_to_add.extend(fragments)
                self.particles.remove(particle)

        # Add new fragments
        self.particles.extend(self.particles_to_add)

        # Clear buffers
        self.particles_to_remove.clear()
        self.particles_to_add.clear()

    def handle_particle_collisions(self):
        """
        Check and resolve all particle to particle collisions.
        """
        # Reset collision flags
        for particle in self.particles:
            particle.in_collision = False
            particle.collision_energy = 0.0

        # Check all pairs of particles
        n = len(self.particles)
        for i in range(n):
            for j in range(i + 1, n):
                if self.detect_collision(self.particles[i], self.particles[j]):
                    self.resolve_collision(self.particles[i], self.particles[j])

    def handle_boundaries(self):
        """
        Handle screen collisions for boundaries
        """
        for particle in self.particles:
            # Bottom boundary
            if particle.position[1] + particle.radius >= self.height:
                particle.position[1] = self.height - particle.radius
                particle.velocity[1] = -particle.velocity[1] * self.restitution

            # Top boundary
            if particle.position[1] - particle.radius <= 0:
                particle.position[1] = particle.radius
                particle.velocity[1] = -particle.velocity[1] * self.restitution

            # Right boundary
            if particle.position[0] + particle.radius >= self.width:
                particle.position[0] = self.width - particle.radius
                particle.velocity[0] = -particle.velocity[0] * self.restitution

            # Left boundary
            if particle.position[0] - particle.radius <= 0:
                particle.position[0] = particle.radius
                particle.velocity[0] = -particle.velocity[0] * self.restitution

    def calculate_total_kinetic_energy(self):
        """Calculate total kinetic energy of all particles."""
        total_ke = 0.0
        for particle in self.particles:
            speed_squared = np.dot(particle.velocity, particle.velocity)
            total_ke += 0.5 * particle.mass * speed_squared
        return total_ke

    def update(self, dt):
        """
        Update simulation by one step at a time.

        Args:
            dt: Time step (seconds)
        """
        # Apply gravity
        self.apply_gravity(dt)

        # Update particle positions
        for particle in self.particles:
            particle.update(dt)

        # Handle particle collisions
        self.handle_particle_collisions()

        # Sprint 3: Process fragmentations
        self.process_fragmentations()

        # Handle boundary collisions
        self.handle_boundaries()

        self.time += dt

    def get_particle_count(self):
        """Return the number of particles in the simulation."""
        return len(self.particles)