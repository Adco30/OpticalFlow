"""Setup script for the OpticalFlow package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="opticalflow",
    version="1.0.0",
    author="OpticalFlow Team",
    author_email="opticalflow@example.com",
    description="Optical Flow Visualization and Optimization System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/opticalflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.5b1",
            "flake8>=3.9.2",
            "mypy>=0.812",
        ],
    },
    entry_points={
        "console_scripts": [
            "opticalflow=app:main",
        ],
    },
    package_data={
        "opticalflow": ["*.md"],
    },
    include_package_data=True,
)
