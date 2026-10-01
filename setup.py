from setuptools import setup

setup(
    name="quantuminversion",
    version="0.1.0",
    description="Fast-inversion block encoding of preconditioned periodic elasticity",
    url="https://github.com/UW-ERSL/QuantumInversion",
    packages=["quantuminversion"],
    python_requires=">=3.10",
    install_requires=["numpy", "scipy"],
    extras_require={"aer": ["qiskit>=1.0", "qiskit-aer"]},
)
