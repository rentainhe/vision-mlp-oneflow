import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vision_mlp_oneflow",
    version="0.0.1",
    author="Ren Tianhe",
    author_email="596106517@qq.com",
    description="vision mlp model based on oneflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rentainhe/vision-mlp-oneflow.git",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)