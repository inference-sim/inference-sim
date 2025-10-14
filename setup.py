from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess, os, tempfile, shutil

class GoBuildInstall(install):
    def run(self):
        # build the Go binary
        here = os.path.dirname(__file__)
        subprocess.run(["go", "build", "-o", "simulation_worker", "main.go"], cwd=here, check=True)
        # move it into the package so it ships
        os.makedirs(os.path.join(here, "inferencesim", "bin"), exist_ok=True)
        shutil.move(os.path.join(here, "simulation_worker"), os.path.join(here, "inferencesim", "bin"))
        super().run()

setup(
    name="inference-sim",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    cmdclass={"install": GoBuildInstall},
    entry_points={"console_scripts": ["simulation-worker=inferencesim.bin:main"]},
)
