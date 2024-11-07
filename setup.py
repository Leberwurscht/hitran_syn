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
        'hapi_initialXsectfork @ git+https://gitlab.com/leberwurscht/hapi_initialXsectfork.git@initialXsectfork',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3'
)
