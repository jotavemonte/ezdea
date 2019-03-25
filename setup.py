import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ezdea",
    version="0.0.1",
    author="João Victor Monte R Tavares",
    author_email="jotavmeonte@gmail.com",
    description="DEA - python interface",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jotavemonte/ezdea",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)