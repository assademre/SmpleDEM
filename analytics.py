"""
Analytics and visualization module for DEM simulation
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os


class SimulationAnalytics:
    """Track and visualize simulation data to see and understand the simulation results"""

    def __init__(self):
        self.time_data = []
        self.particle_count = []
        self.collision_count = []
        ## Might use different approcah for these ones
        self.energy_data = []
        self.fragments_data = []
        self.frames = []

    def record_frame(self, sim):
        """Record simulation"""
        self.time_data.append(sim.time)
        self.particle_count.append(sim.get_particle_count())
        self.collision_count.append(sim.total_collisions)
        self.fragments_data.append(sim.total_fragments_created)

        if len(sim.collision_energies) > 0:
            avg_energy = np.mean(sim.collision_energies[-10:]) if len(sim.collision_energies) >= 10 else np.mean(
                sim.collision_energies)
        else:
            avg_energy = 0
        self.energy_data.append(avg_energy)

    def generate_graphs(self, output_dir="sim_results"):
        """Generate analysis graphs"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('DEM Simulation Analysis', fontsize=16, fontweight='bold')

        # Particle count over time plot
        axes[0, 0].plot(self.time_data, self.particle_count, linewidth=2, color='#64C8FF')
        axes[0, 0].set_xlabel('Time (s)', fontsize=11)
        axes[0, 0].set_ylabel('Particle Count', fontsize=11)
        axes[0, 0].set_title('Particles Over Time', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].fill_between(self.time_data, self.particle_count, alpha=0.2, color='#64C8FF')

        # Cumulative colusion plot
        axes[0, 1].plot(self.time_data, self.collision_count, linewidth=2, color='#FF6464')
        axes[0, 1].set_xlabel('Time (s)', fontsize=11)
        axes[0, 1].set_ylabel('Total Collisions', fontsize=11)
        axes[0, 1].set_title('Cumulative Collisions', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].fill_between(self.time_data, self.collision_count, alpha=0.2, color='#FF6464')

        # Collison energy plot
        axes[1, 0].plot(self.time_data, self.energy_data, linewidth=2, color='#FFD700')
        axes[1, 0].set_xlabel('Time (s)', fontsize=11)
        axes[1, 0].set_ylabel('Avg Energy (J)', fontsize=11)
        axes[1, 0].set_title('Average Collision Energy', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].fill_between(self.time_data, self.energy_data, alpha=0.2, color='#FFD700')

        # Total fragments plot
        axes[1, 1].plot(self.time_data, self.fragments_data, linewidth=2, color='#64FF64')
        axes[1, 1].set_xlabel('Time (s)', fontsize=11)
        axes[1, 1].set_ylabel('Fragments Created', fontsize=11)
        axes[1, 1].set_title('Cumulative Fragment Creation', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(self.time_data, self.fragments_data, alpha=0.2, color='#64FF64')

        plt.tight_layout()

        # Save graph
        graph_path = os.path.join(output_dir, f"analysis_{timestamp}.png")
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        print(f"Graph saved to: {graph_path}")
        plt.close()

        return graph_path


    def generate_summary_report(self, sim, output_dir="sim_results"):
        """Generate text summary report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"report_{timestamp}.txt")

        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("DEM FRAGMENTATION SIMULATION - FINAL REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Simulation Duration: {sim.time:.2f} seconds\n")
            f.write(f"Final Particle Count: {sim.get_particle_count()}\n")
            f.write(f"Total Collisions: {sim.total_collisions}\n")
            f.write(f"Total Fragments Created: {sim.total_fragments_created}\n\n")

            f.write("Energy Statistics:\n")
            f.write("-" * 60 + "\n")
            if len(sim.collision_energies) > 0:
                f.write(f"Average Collision Energy: {np.mean(sim.collision_energies):.2f} J\n")
                f.write(f"Max Collision Energy: {np.max(sim.collision_energies):.2f} J\n")
                f.write(f"Min Collision Energy: {np.min(sim.collision_energies):.2f} J\n")
                f.write(f"Std Dev: {np.std(sim.collision_energies):.2f} J\n")
            else:
                f.write("No collisions recorded\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("SIMULATION PARAMETERS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Energy Threshold: {sim.energy_threshold:.1f} J\n")
            f.write(f"Min Fragment Radius: {sim.min_fragment_radius:.1f}\n")
            f.write(f"Fragments per Break: {sim.num_fragments}\n")
            f.write(f"Max Particles: {sim.max_particles}\n")
            f.write(f"Gravity: {sim.gravity / 100:.2f} m/s^2\n")

        print(f"Report saved to: {report_path}")
        return report_path


def create_gif_from_frames(frames, output_dir="sim_results", fps=30):
    """
    Create animated GIF from pygame frames

    Usage in main.py:
        from PIL import Image
        analytics.frames.append(pygame.surfarray.array3d(screen))
        # After simulation ends:
        create_gif_from_frames(analytics.frames)
    """
    try:
        from PIL import Image

        if not frames:
            print("No frames to create GIF")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(output_dir, f"simulation_{timestamp}.gif")

        # Convert frames to images
        pil_frames = []
        for frame in frames:
            img = Image.fromarray(frame.astype('uint8'))
            pil_frames.append(img)

        # Save as GIF
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=1000 // fps,
            loop=0
        )
        print(f"GIF saved to: {gif_path}")
        return gif_path

    except ImportError:
        print("Package error")
        return None