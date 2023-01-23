from pathlib import Path
from setuptools import find_packages, setup

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name="nuance",
    version="0.0.2",
    author="Lionel J. Garcia",
    packages=["nuance"],
    description="Transit detection in correlated noises",
    # packages=find_packages(exclude=["test"]),
    # include_package_data = True,
    license="MIT",
    url="https://github.com/lgrcia/nuance",
    # entry_points="""
    #     [console_scripts]
    #     prose=main:cli
    # """,
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "jax", "jaxopt", "tinygp", "tqdm"],
    extras_require={
        "dev": [
            "sphinx",
            "nbsphinx",
            "docutils",
            "jupyterlab",
            "myst-parser",
            "twine",
            "sphinx-book-theme @ git+https://github.com/executablebooks/sphinx-book-theme.git@0c404524aed8f1e0009fc03a0f7e04282fd238ab#egg=sphinx-book-theme",
            "black",
            "myst_nb",
            "sphinx-copybutton"
        ]
    },
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
