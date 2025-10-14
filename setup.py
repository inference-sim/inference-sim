from setuptools import setup
from setuptools.command.install import install
import subprocess

class GoBuildCommand(install):
    """Custom install command to build the Go binary automatically."""
    def run(self):
        print("🏗️  Building Go binary (simulation_worker)...")
        subprocess.run(["go", "build", "-o", "simulation_worker", "main.go"], check=True)
        print("✅  Go binary built successfully.")
        super().run()

setup(
    name="inference_sim",
    version="0.1.0",
    py_modules=[
        "request_rate_sweep",
        "experiment_constants",
        "generate_random_prompts",
    ],
    install_requires=["pandas"],
    cmdclass={"install": GoBuildCommand},
)