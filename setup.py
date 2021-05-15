from setuptools import find_packages, setup


def readme(path: str = "./README.md"):
    with open(path) as f:
        return f.read()


setup(
    name="nemesys",
    version="0.1.0.dev1",
    description="Neural Memory System",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/suflaj/nemesys",
    author="Miljenko Šuflaj",
    author_email="headsouldev@gmail.com",
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Environment :: GPU",
        "Framework :: Flake8",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="neural memory system nemesys",
    project_urls={
        "Source": "https://github.com/suflaj/nemesys",
        "Issues": "https://github.com/suflaj/nemesys/issues",
    },
    packages=find_packages(
        include=(
            "nemesys",
            "nemesys.*",
        )
    ),
    install_requires=[
        "torch",
        "tqdm",
    ],
    python_requires=">=3.6, <3.9",
    package_data={
        "documentation": ["docs/*"],
        "demonstration": ["demo/*"],
    },
    include_package_data=True,
    zip_safe=False,
)
