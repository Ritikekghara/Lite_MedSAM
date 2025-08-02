from setuptools import find_packages, setup
import os

try:
    with open(os.path.join(os.path.dirname(__file__), "README.md"), encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "A lightweight version of MedSAM for medical image segmentation."


install_requires = [
    "numpy",
    "torch",
    "timm",
    "Pillow",
    "scipy",
    "scikit-image",
    "SimpleITK>=2.2.1",
    "monai",
    "opencv-python",
]

extras_require = {
    "app": [
        "flask",
        "waitress",
    ],
    "notebook": [
        "jupyterlab",
        "ipywidgets",
        "ipympl",
        "matplotlib",
        "tqdm",
        "pandas",
        "nibabel",
    ],
    "onnx": [
        "pycocotools",
        "onnx",
        "onnxruntime",
    ],
    "dev": [
        "flake8",
        "isort",
        "black",
        "mypy",
    ],
}

all_extras = []
for extra_deps in extras_require.values():
    all_extras.extend(extra_deps)
extras_require["all"] = sorted(list(set(all_extras)))


setup(
    name="litemedsam",
    version="0.0.1",
    author="Ritik Kumar",
    author_email="ritikekghara@gmail.com",
    description="A lightweight version of MedSAM for medical image segmentation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0", 
    python_requires=">=3.9",
    packages=find_packages(exclude=["notebooks", "tests"]),
    install_requires=install_requires,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)