from KinematicSolver import GravitySolver
from TNGloading import loadGalaxy, center, faceon
from AutoGMM import AutoGMM
from Tools import cal_structure_info
from Visualize import PhaseSpace

run = 'TNG100-3'
basePath = f"/Users/yuwa/sim.TNG/{run}/output"
subID = 1
snapNum = 99
galaxy = loadGalaxy(basePath, run, snapNum, subID)
center(galaxy)
faceon(galaxy)
galaxy = GravitySolver(galaxy, Solver='Tree')

AutoGMM_model = AutoGMM(galaxy, n_components=2)
#PhaseSpace(AutoGMM_model.X)

GMM_model = AutoGMM_model.fit()
Particles_data, GMM_info = AutoGMM_model.decompose()

Structure_info = cal_structure_info(galaxy, Particles_data)

