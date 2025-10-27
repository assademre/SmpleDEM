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
        self.size_distributions = []  # Store particle sizes over time

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

        # Record particle sizes for distribution
        particle_sizes = [p.radius for p in sim.particles]
        self.size_distributions.append(particle_sizes)

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

    def generate_psd_analysis(self, output_dir="sim_results"):
        """Generate Particle Size Distribution analysis"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if not self.size_distributions:
            print("No size distribution data available")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create figure with PSD plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Particle Size Distribution Analysis', fontsize=16, fontweight='bold')

        # Get initial, middle, and final distributions
        initial_sizes = self.size_distributions[0] if len(self.size_distributions) > 0 else []
        middle_idx = len(self.size_distributions) // 2
        middle_sizes = self.size_distributions[middle_idx] if len(self.size_distributions) > middle_idx else []
        final_sizes = self.size_distributions[-1] if len(self.size_distributions) > 0 else []

        # Plot 1: Size distribution histogram comparison
        if initial_sizes:
            axes[0, 0].hist(initial_sizes, bins=15, alpha=0.5, label='Initial', color='#64C8FF', edgecolor='black')
        if middle_sizes:
            axes[0, 0].hist(middle_sizes, bins=15, alpha=0.5, label='Mid-sim', color='#FFD700', edgecolor='black')
        if final_sizes:
            axes[0, 0].hist(final_sizes, bins=15, alpha=0.5, label='Final', color='#FF6464', edgecolor='black')
        axes[0, 0].set_xlabel('Particle Radius', fontsize=11)
        axes[0, 0].set_ylabel('Frequency', fontsize=11)
        axes[0, 0].set_title('Size Distribution Evolution', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Cumulative size distribution (CSD)
        if final_sizes:
            sorted_sizes = np.sort(final_sizes)
            cumulative = np.arange(1, len(sorted_sizes) + 1) / len(sorted_sizes) * 100
            axes[0, 1].plot(sorted_sizes, cumulative, linewidth=2, color='#9370DB')
            axes[0, 1].set_xlabel('Particle Radius', fontsize=11)
            axes[0, 1].set_ylabel('Cumulative Passing (%)', fontsize=11)
            axes[0, 1].set_title('Cumulative Size Distribution', fontweight='bold')
            axes[0, 1].grid(True, alpha=0.3)

            # Add d50 line (median size)
            d50_idx = np.argmin(np.abs(cumulative - 50))
            d50 = sorted_sizes[d50_idx]
            axes[0, 1].axhline(y=50, color='red', linestyle='--', alpha=0.5, label=f'd50 = {d50:.2f}')
            axes[0, 1].axvline(x=d50, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].legend()

        # Plot 3: Mean size over time
        mean_sizes = [np.mean(sizes) if sizes else 0 for sizes in self.size_distributions]
        axes[1, 0].plot(self.time_data, mean_sizes, linewidth=2, color='#32CD32')
        axes[1, 0].set_xlabel('Time (s)', fontsize=11)
        axes[1, 0].set_ylabel('Mean Particle Radius', fontsize=11)
        axes[1, 0].set_title('Mean Size Evolution', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].fill_between(self.time_data, mean_sizes, alpha=0.2, color='#32CD32')

        # Plot 4: Size distribution statistics over time
        std_sizes = [np.std(sizes) if sizes else 0 for sizes in self.size_distributions]
        min_sizes = [np.min(sizes) if sizes else 0 for sizes in self.size_distributions]
        max_sizes = [np.max(sizes) if sizes else 0 for sizes in self.size_distributions]

        axes[1, 1].plot(self.time_data, max_sizes, linewidth=2, label='Max', color='#FF6464')
        axes[1, 1].plot(self.time_data, mean_sizes, linewidth=2, label='Mean', color='#32CD32')
        axes[1, 1].plot(self.time_data, min_sizes, linewidth=2, label='Min', color='#64C8FF')
        axes[1, 1].fill_between(self.time_data, min_sizes, max_sizes, alpha=0.1, color='gray')
        axes[1, 1].set_xlabel('Time (s)', fontsize=11)
        axes[1, 1].set_ylabel('Particle Radius', fontsize=11)
        axes[1, 1].set_title('Size Range Evolution', fontweight='bold')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Save PSD graph
        psd_path = os.path.join(output_dir, f"psd_analysis_{timestamp}.png")
        plt.savefig(psd_path, dpi=150, bbox_inches='tight')
        print(f"PSD analysis saved to: {psd_path}")
        plt.close()

        return psd_path

    def calculate_psd_metrics(self):
        """Calculate key PSD metrics for the final distribution"""
        if not self.size_distributions or not self.size_distributions[-1]:
            return {}

        final_sizes = np.array(self.size_distributions[-1])
        sorted_sizes = np.sort(final_sizes)

        # Calculate percentiles (d10, d50, d90)
        d10 = np.percentile(sorted_sizes, 10)
        d50 = np.percentile(sorted_sizes, 50)  # median
        d90 = np.percentile(sorted_sizes, 90)

        # Calculate uniformity coefficient
        d60 = np.percentile(sorted_sizes, 60)
        uniformity = d60 / d10 if d10 > 0 else 0

        # Calculate span (measure of distribution width)
        span = (d90 - d10) / d50 if d50 > 0 else 0

        metrics = {
            'mean_size': np.mean(final_sizes),
            'std_dev': np.std(final_sizes),
            'min_size': np.min(final_sizes),
            'max_size': np.max(final_sizes),
            'd10': d10,
            'd50': d50,
            'd90': d90,
            'uniformity_coefficient': uniformity,
            'span': span,
            'total_particles': len(final_sizes)
        }

        return metrics

    def generate_summary_report(self, sim, output_dir="sim_results"):
        """Generate text summary report"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"report_{timestamp}.txt")

        # Calculate PSD metrics
        psd_metrics = self.calculate_psd_metrics()

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

            # Add PSD metrics section
            if psd_metrics:
                f.write("\n" + "=" * 60 + "\n")
                f.write("PARTICLE SIZE DISTRIBUTION METRICS:\n")
                f.write("-" * 60 + "\n")
                f.write(f"Mean Particle Size: {psd_metrics['mean_size']:.2f}\n")
                f.write(f"Std Deviation: {psd_metrics['std_dev']:.2f}\n")
                f.write(f"Size Range: {psd_metrics['min_size']:.2f} - {psd_metrics['max_size']:.2f}\n")
                f.write(f"d10 (10th percentile): {psd_metrics['d10']:.2f}\n")
                f.write(f"d50 (median): {psd_metrics['d50']:.2f}\n")
                f.write(f"d90 (90th percentile): {psd_metrics['d90']:.2f}\n")
                f.write(f"Uniformity Coefficient: {psd_metrics['uniformity_coefficient']:.2f}\n")
                f.write(f"Span: {psd_metrics['span']:.2f}\n")

            f.write("\n" + "=" * 60 + "\n")
            f.write("SIMULATION PARAMETERS:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Energy Threshold: {sim.energy_threshold:.1f} J\n")
            f.write(f"Min Fragment Radius: {sim.min_fragment_radius:.1f}\n")
            f.write(f"Fragments per Break: {sim.num_fragments}\n")
            f.write(f"Max Particles: {sim.max_particles}\n")
            f.write(f"Gravity: {sim.gravity / 100:.2f} m/s^2\n")
            f.write(f"\nEnergy Conservation:\n")
            f.write(f"Total energy lost to fragmentation: {sim.energy_lost_to_fragmentation:.2f} J\n")

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