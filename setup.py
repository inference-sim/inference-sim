from setuptools import setup
from setuptools.command.install import install
import subprocess
import os
import shutil


class GoBuildCommand(install):
    """Builds Go binary and places it next to run_blis_sweep.py."""
    def run(self):
        print("Building Go binary (simulation_worker)...")
        subprocess.run(["go", "build", "-o", "simulation_worker", "main.go"], check=True)

        # Continue normal Python install
        super().run()

        # Copy binary into the installed site-packages folder (next to run_blis_sweep.py)
        site_packages_dir = self.install_lib
        dest = os.path.join(site_packages_dir, "simulation_worker")
        print(f"Copying Go binary to {dest}")
        shutil.copy("simulation_worker", dest)

        print("Go binary built and installed successfully.")


setup(
    name="inference_sim",
    version="0.1.0",
    description="Simulation sweep utilities with Go backend",
    py_modules=[
        "run_blis_sweep",
        "experiment_constants_inference",
        "generate_random_prompts",
    ],
    install_requires=[
        "pandas>=2.3.1",
        "bayesian-optimization>=3.0.1",
        "ipykernel>=6.29.5",
        "ipywidgets>=8.1.7",
        "matplotlib>=3.9.4",
        "nbformat>=5.10.4",
        "optuna>=4.4.0",
        "optunahub>=0.3.1",
        "pandas-profiling>=3.2.0",
        "plotly>=6.2.0",
        "transformers>=4.53.0",
        "widgetsnbextension>=4.0.14",
    ],
    cmdclass={"install": GoBuildCommand},
    include_package_data=True,
    python_requires=">=3.9",
)