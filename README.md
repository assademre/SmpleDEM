# Simplified DEM Fragmentation Simulation (SmpleDEM)

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
E = 0.5 * m_reduced * v_relative^2
```
where `m_reduced = (m1 * m2) / (m1 + m2)`

### Fragmentation Rules
A particle breaks only if:
1. Collision energy > energy threshold (default 1000 J)
2. Resulting fragment radius >= minimum fragment radius (default 5.0)

### Fragment Generation
When a particle breaks:
- Mass is conserved between fragments
- Fragments spread radially from the original position
- Each fragment keeps parent velocity plus a small outward push
- 0.7 efficiency factor represents energy loss to fracture

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
- `analysis_[timestamp].png` – combined trend graphs
- `report_[timestamp].txt` – summary text
- `simulation_[timestamp].gif` – optional animation

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
