"""
DEM Fragmentation Simulation
"""
import pygame
import numpy as np
from simulation import DEMSimulation
from particle import Particle


# Constants
WIDTH:int = 800
HEIGHT:int = 600
FPS:int = 60
BACKGROUND_COLOR:tuple = (20, 20, 30)
PARTICLE_COLOR:tuple = (100, 200, 255)
COLLISION_COLOR:tuple = (255, 100, 100)  # Color red
GRAVITY:float = 9.81
STARTING_PARTICLES = 2
MAX_PARTICLES = 300


def main():
    """Main simulation loop to see"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Smple DEM Sim")
    clock = pygame.time.Clock()
    
    # Initialize simulation
    sim = DEMSimulation(width=WIDTH, height=HEIGHT, gravity=GRAVITY)

    # sim.add_particle(Particle(x=200, y=50, vx=50, vy=0, radius=15, density=1.0))
    # sim.add_particle(Particle(x=400, y=100, vx=-30, vy=0, radius=20, density=1.0))
    # sim.add_particle(Particle(x=600, y=80, vx=-20, vy=50, radius=12, density=1.0))
    # sim.add_particle(Particle(x=300, y=150, vx=40, vy=20, radius=25, density=1.0))

    # Addding particles
    sim.max_particles = MAX_PARTICLES
    print(f"Creating {STARTING_PARTICLES} initial particles...")
    for i in range(STARTING_PARTICLES):
        x = np.random.randint(50, WIDTH - 50)
        y = np.random.randint(50, 150)
        vx = np.random.uniform(-50, 50)
        vy = np.random.uniform(0, 50)
        radius = np.random.randint(10, 25)
        sim.add_particle(Particle(x, y, vx, vy, radius, density=1.0))
    
    # Main loop
    running = True
    dt = 1.0 / FPS

    info_font = pygame.font.Font(None, 24)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    x = np.random.randint(50, WIDTH - 50)
                    y = np.random.randint(50, 150)
                    vx = np.random.uniform(-50, 50)
                    vy = np.random.uniform(0, 50)
                    radius = np.random.randint(10, 25)
                    sim.add_particle(Particle(x, y, vx, vy, radius, density=1.0))

        sim.update(dt)

        screen.fill(BACKGROUND_COLOR)
        
        # Draw particles
        for particle in sim.particles:
            pos = (int(particle.position[0]), int(particle.position[1]))

            if hasattr(particle, 'in_collision') and particle.in_collision:
                energy = min(particle.collision_energy, 1000.0) if hasattr(particle, 'collision_energy') else 0
                intensity = int(255 * (energy / 1000.0))
                color = (255, 255 - intensity, 255 - intensity)
            else:
                color = PARTICLE_COLOR
            pygame.draw.circle(screen, color, pos, int(particle.radius))
            pygame.draw.circle(screen, (255, 255, 255), pos, 2)  # center dot
        
        # Info text
        info_text = info_font.render(f"Particles: {sim.get_particle_count()}/{sim.max_particles}", True, (255, 255, 255))
        screen.blit(info_text, (10, 10))
        
        time_text = info_font.render(f"Time: {sim.time:.1f}s", True, (255, 255, 255))
        screen.blit(time_text, (10, 35))

        if hasattr(sim, 'total_collisions'):
            collisions_text = info_font.render(f"Collisions: {sim.total_collisions}", True, (255, 255, 255))
            screen.blit(collisions_text, (10, 60))

        if hasattr(sim, 'total_fragments_created'):
            fragments_text = info_font.render(f"Fragments Created: {sim.total_fragments_created}", True,
                                              (100, 255, 100))
            screen.blit(fragments_text, (10, 85))

        if hasattr(sim, 'collision_energies') and len(sim.collision_energies) > 0:
            avg_energy = np.mean(sim.collision_energies[-10:])
            energy_text = info_font.render(f"Avg Energy: {avg_energy:.1f} J", True, (255, 255, 100))
            screen.blit(energy_text, (10, 110))

        help_text = info_font.render("SPACE: Add particle, ESC: Quit", True, (150, 150, 150))
        screen.blit(help_text, (10, HEIGHT - 30))

        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    print(f"\nSimulation Summary")
    print(f"Total particles: {sim.get_particle_count()}")
    if hasattr(sim, 'total_collisions'):
        print(f"Total collisions: {sim.total_collisions}")
    if hasattr(sim, 'collision_energies') and len(sim.collision_energies) > 0:
        print(f"Average collision energy: {np.mean(sim.collision_energies):.2f} J")
        print(f"Max collision energy: {np.max(sim.collision_energies):.2f} J")


if __name__ == "__main__":
    main()
