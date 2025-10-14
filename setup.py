from setuptools import setup
from setuptools.command.install import install
import subprocess, os, shutil

class GoBuildInstall(install):
    def run(self):
        here = os.path.abspath(os.path.dirname(__file__))
        # --- Build Go binary ---
        subprocess.run(["go", "build", "-o", "simulation_worker", "main.go"], cwd=here, check=True)

        # --- Move Go binary into bin/ so it's installed ---
        os.makedirs(os.path.join(here, "bin"), exist_ok=True)
        shutil.move(os.path.join(here, "simulation_worker"), os.path.join(here, "bin"))

        super().run()

setup(
    name="inferencesim",              # package name for pip
    version="0.1.0",
    py_modules=[
        "request_rate_sweep",
        "experiment_constants",
        "generate_random_prompts"
    ],
    include_package_data=True,
    cmdclass={"install": GoBuildInstall},
    entry_points={
        "console_scripts": [
            "inferencesim-sweep=request_rate_sweep:main",  # optional CLI shortcut
        ],
    },
)
