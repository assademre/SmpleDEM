# Simplified DEM Fragmentation Simulation (SmpleDEM)

- [Overview](#overview)
- [Features](#features)
  - [Particle Physics](#particle-physics)
  - [Collision Detection](#collision-detection)
  - [Energy-Based Fragmentation](#energy-based-fragmentation)
  - [Physical Constraints](#physical-constraints)
  - [Real-Time Visualization](#real-time-visualization)
  - [Data Analytics](#data-analytics)
  - [Logging](#logging)
- [Physics Model](#physics-model)
  - [Collision Energy Calculation](#collision-energy-calculation)
  - [Fragmentation Rules](#fragmentation-rules)
  - [Fragment Generation](#fragment-generation)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setup](#setup)
- [Results](#results)
  - [Sample Simulation Output](#sample-simulation-output)
  - [Particle Size Distribution Analysis](#particle-size-distribution-analysis)
  - [Physical Behavior](#physical-behavior)
- [Usage](#usage)
  - [Running](#running)
  - [Configuration](#configuration)
  - [Controls](#controls)
  - [Output](#output)
- [Project Structure](#project-structure)
- [Limitations & Design Choices](#limitations--design-choices)
  - [2D Only](#2d-only)
  - [Fragmentation Model](#fragmentation-model)
  - [Energy Threshold](#energy-threshold)
  - [No Surface Wear](#no-surface-wear)
- [Performance](#performance)
- [Future Work](#future-work)
- [References](#references)
- [Notes](#notes)

2D Discrete Element Method (DEM) simulation with energy based particle breakage for studying comminution and fragmentation behavior.

## Overview

This is a small DEM sandbox where particles fall, collide, and sometimes break apart if the impact energy is high enough. The goal is to keep it simple but still show how fragmentation works in granular systems. It’s not meant to be physically perfect, just a clean framework to experiment with breakage and see some patterns.

The simulation tracks collisions, energies, and fragments in real time. After running, it creates a few plots and a short report for analysis.

## Features

- **Particle Physics**: position, velocity, mass, and radius tracking with gravity
- **Collision Detection**: pairwise collision checking with elastic response
- **Energy-Based Fragmentation**: particles break when impact energy goes above a threshold
- **Physical Constraints**:
  - Minimum fragment radius to stop infinite fragmentation
  - Particle cap to prevent slowdown
  - Partial energy conservation (not fully working yet)
- **Real-Time Visualization**: color intensity based on collision energy
- **Data Analytics**: plots for particle count, collisions, energy, and fragment creation
- **Logging**: simple reports for collisions and breakage events

## Physics Model

### Collision Energy Calculation
```
E = 0.5 * m1 * v1² + 0.5 * m2 * v2²
```
where `m_reduced = (m1 * m2) / (m1 + m2)`
### Fragmentation Rules
A particle breaks only if:
1. Collision energy > energy threshold (default 1000 J)
2. Resulting fragment radius >= minimum fragment radius (default 5.0)

This ensures physically reasonable breakage and prevents the creation of unrealistically small particles.

### Fragment Generation
When a particle breaks:
- **Mass conservation**: Total fragment mass equals parent mass
- **Radial distribution**: Fragments spawn around parent position in radial pattern
- **Velocity inheritance**: Each fragment inherits parent velocity plus radial velocity component from collision energy
- **Fragmentation efficiency**: 70% of kinetic energy preserved; 30% lost to bond-breaking

The fragment radius is calculated to conserve area:
```
r_fragment = √(A_parent / (n_fragments * π))
```
where fragments are only created if `r_fragment ≥ min_fragment_radius`.

## Installation

### Requirements
- Python 3.8+
- pygame
- numpy
- matplotlib
- pillow

### Setup
```bash
pip install -r requirements.txt
```

## Results

### Sample Simulation Output

A typical simulation run with the following parameters demonstrates clear comminution behavior:
- Initial particles: 3 particles (radius 20 to 30)
- Energy threshold: 800 J???
- Fragments per break: 3
- Simulation time: 30 seconds

**Key Observations:**
- Particle count increased from 3 to ~85 particles
- Mean particle size (d50) decreased from 25.4 to 6.8 (73% reduction)
- Total collisions: ~450, with ~60 fragmentation events
- Size distribution transitioned from narrow (few large particles) to broad (many small fragments)

### Particle Size Distribution Analysis

The PSD analysis reveals typical grinding behavior:
- **d10**: Decreased from 22 to 3.2 (smallest 10% of particles)
- **d50**: Decreased from 25 to 6.8 (median particle size)
- **d90**: Decreased from 28 to 12.5 (largest 10% still relatively big)
- **Span**: Increased from 0.24 to 1.37, (indicating broader size distribution post fragmentation

The cumulative size distribution curve shows the characteristic S-shape expected in comminution processes, with most particles concentrated in the fine fraction after extended grinding.

### Physical Behavior

The simulation correctly captures:
- Energy-dependent breakage (low energy collisions don't ve fragment)
- Cascade fragmentation (large to medium, medium to small particles)
- Energy dissipation through collisions and fragmentation
- Particle dynamics under gravity with realistic bouncing and rolling

## Usage

### Running
```bash
python main.py
```

### Configuration
In `main.py`:
```python
STARTING_PARTICLES = 2
MAX_PARTICLES = 300
NUM_FRAGMENTS = 2
ENERGY_THRESHOLD = 1000.0
RECORD_GIF = False
```
In `simulation.py`:
```python
self.min_fragment_radius = 5.0
```

### Controls
- **SPACE**: add a random particle
- **ESC**: quit simulation

### Output
After finishing:
1. **analysis_[timestamp].png** - Panles showing:
   - Particle count evolution over time
   - Cumulative collision events
   - Average collision energy
   - Total fragments created

2. **psd_analysis_[timestamp].png** - Particle size distribution analysis:
   - Size distribution histogram (initial vs mid simulation vs final)
   - Cumulative size distribution with d50 marker
   - Mean particle size evolution over time
   - Size range evolution (min/mean/max bands)

3. **report_[timestamp].txt** - Detailed summary including:
   - Simulation duration and final statistics
   - Energy statistics (mean, max, min, std dev)
   - PSD metrics (d10, d50, d90, uniformity coefficient, span)
   - Simulation parameters used

4. **simulation_[timestamp].gif** - Recording of the session

All files are timestamped to avoid overwriting previous runs.

## Project Structure
```
.
├── main.py           # Simulation and rendering
├── simulation.py     # DEM physics and logic
├── particle.py       # Particle class
├── analytics.py      # Data tracking and plotting
└── README.md
```

## Limitations & Design Choices

### 2D Only
This is 2D for simplicity. 3D would be more realistic for stress and crack propagation but much slower and harder to debug. Especially in Python

### Fragmentation Model
Particles split evenly. Real fractures depend on microstructure, crack direction, and local stress, but this version keeps it predictable and stable.

### Energy Threshold
The threshold is global. Real systems depend on material, geometry, and impact location.

### No Surface Wear
Particles only break. No slow surface wear or deformation.

## Performance

- Collision detection is O(n^2), so >300 particles gets slow
- Fragment control through caps and radius limits
- GIF recording off by default to save memory

Possible optimizations later:
- Spatial hashing for faster collisions
- GPU acceleration
- Smaller timestep during high energy collisions
- Using Cython or C++ support

## Future Work

- 3D version
- Material dependent breakage
- Real fracture surfaces
- Parallel computation
- Experimental data comparison
- Integration with grinding simulation workflows

## References

- Cundall & Strack (1979) – DEM fundamentals
- Tavarez & Plesha (2007), Moreno-Atanasio et al. (2006) – particle breakage in DEM
- Ben-Nun & Aharonov (2006) – for energy based fragmentation concept 

This is just a proof of concept. Real DEM codes use more advanced stress based and damage models.

## Notes

- Coefficient of restitution: 0.8
- Gravity scaled by 100
- Particles bounce on walls (will be changed with material dependent breakage feature)
- Collisions above 100 J are logged

---

Built as a small side project to explore particle breakage in DEM.
