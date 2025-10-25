"""Main simulation engine for DEM
"""
import numpy as np
from particle import Particle
from collections import defaultdict


class SpatialHashGrid:
    """
    Spatial hash grid for fast neighbor finding.
    """

    def __init__(self, cell_size):
        """
        Initialize spatial hash grid.

        Args:
            cell_size: Size of each grid cell (should be ~2x max particle radius)
        """
        self.cell_size = cell_size
        self.grid = defaultdict(list)

    def clear(self):
        """Clear the grid for next frame"""
        self.grid.clear()

    def _hash(self, x, y):
        """Convert world position to grid cell coordinates"""
        cell_x = int(x // self.cell_size)
        cell_y = int(y // self.cell_size)
        return cell_x, cell_y

    def insert(self, particle):
        """Insert particle into grid"""
        cell = self._hash(particle.position[0], particle.position[1])
        self.grid[cell].append(particle)

    def get_nearby_particles(self, particle):
        """
        Get all particles in neighboring cells (including same cell).
        Only checks 9 cells instead of all particles.
        """
        px, py = particle.position
        cell_x, cell_y = self._hash(px, py)

        nearby = []
        # Check 3x3 neighborhood around particle
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor_cell = (cell_x + dx, cell_y + dy)
                if neighbor_cell in self.grid:
                    nearby.extend(self.grid[neighbor_cell])

        return nearby


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
        self.energy_threshold = 15000.0  # Energy required to break a particle
        self.min_fragment_radius = 3.0  # Minimum size before particles can't break
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

    def detect_collision(self, p1, p2):
        """
        Check if two particles are colliding.
        Args:
            p1, p2: Particle objects

        Returns:
            bool: True if particles overlap
        """
        # Bounding box check. Seems like the cheapest option
        dx = abs(p1.position[0] - p2.position[0])
        dy = abs(p1.position[1] - p2.position[1])
        sum_radii = p1.radius + p2.radius

        # Early exit if definitely not colliding
        if dx > sum_radii or dy > sum_radii:
            return False

        # Only calculate expensive distance if needed
        distance_squared = dx*dx + dy*dy
        return distance_squared < sum_radii * sum_radii

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
        # # using kinetic energy instead
        # ke1 = 0.5 * p1.mass * np.linalg.norm(p1.velocity) ** 2
        # ke2 = 0.5 * p2.mass * np.linalg.norm(p2.velocity) ** 2
        # impact_energy = ke1 + ke2

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
            fragment_area = (np.pi * p1.radius ** 2) / self.num_fragments
            min_fragment_radius_calc = np.sqrt(fragment_area / np.pi)

            if min_fragment_radius_calc >= self.min_fragment_radius:
                self.mark_for_fragmentation(p1, impact_energy)
                fragmented = True
            if min_fragment_radius_calc >= self.min_fragment_radius:
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
        fragmentation_efficiency = 0.7  # kinetic energy preservation

        # Calculate how much energy to add from the collision
        collision_energy = getattr(particle, 'fragment_energy', 0)
        available_energy = collision_energy * fragmentation_efficiency
        energy_per_fragment = available_energy / self.num_fragments

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
        if not self.particles_to_remove:
            return

        # Build set for O(1) lookup instead of O(n) list operations
        to_remove_set = set(self.particles_to_remove)

        # Create all fragments first
        new_fragments = []
        for particle in self.particles_to_remove:
            if particle in self.particles:
                fragments = self.fragment_particle(particle)
                new_fragments.extend(fragments)

        # Filter particles in one operation
        self.particles = [p for p in self.particles if p not in to_remove_set]

        # Add new fragments
        self.particles.extend(new_fragments)

        # Clear buffers
        self.particles_to_remove.clear()
        self.particles_to_add.clear()

    def handle_particle_collisions(self):
        """
        Check and resolve all particle-to-particle collisions.
        """
        # Reset collision flags
        for particle in self.particles:
            particle.in_collision = False
            particle.collision_energy = 0.0

        # Early exit if too few particles
        if len(self.particles) < 2:
            return

        # Build spatial hash grid
        max_radius = max((p.radius for p in self.particles), default=10)
        cell_size = max_radius * 2.5

        grid = SpatialHashGrid(cell_size)
        for particle in self.particles:
            grid.insert(particle)

        # Check collisions only within nearby cells
        checked_pairs = set()

        for particle in self.particles:
            nearby = grid.get_nearby_particles(particle)

            for other in nearby:
                # Skip self
                if particle is other:
                    continue

                # Avoid checking same pair twice using unique IDs
                pair = tuple(sorted([id(particle), id(other)]))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)

                # Check collision
                if self.detect_collision(particle, other):
                    self.resolve_collision(particle, other)

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

        # Process fragmentations
        self.process_fragmentations()

        # Handle boundary collisions
        self.handle_boundaries()

        self.time += dt

    def get_particle_count(self):
        """Return the number of particles in the simulation."""
        return len(self.particles)