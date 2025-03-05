from setuptools import setup, find_packages

setup(
    name="your_project_name",
    version="0.1.0",
    author="Your Name",
    description="Your project description",
    packages=find_packages(),  # Automatically finds all packages
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires=">=3.7"
)