import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setuptools.setup(
    name="hitran_syn",
    version=version,
    author="Leberwurscht",
    author_email="leberwurscht@hoegners.de",
    description="complex-valued absorption spectra from HITRAN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/leberwurscht/hitran_syn",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy>=1.4.0',
        'hitran-api @ git+https://github.com/Leberwurscht/hapi.git@initial_Xsect#egg=hitran-api-1.2.2.1initialXsect',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3'
)
