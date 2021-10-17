import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "ANN"
USER_NAME = "DARSHANH-analytics"

setuptools.setup(
    name=f"{PROJECT_NAME}-{USER_NAME}",
    version="0.0.1",
    author=USER_NAME,
    author_email="darshan8983@gmail.com",
    description="A small package of ANN Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",

    # commenting project_urls,classifiers since we are not publishing the package available online to pypi website

    # project_urls={
    #    "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    # },

    # classifiers=[
    #     "Programming Language :: Python :: 3",
    #     "License :: OSI Approved :: MIT License",
    #     "Operating System :: OS Independent",
    # ],

    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "tensorflow"
    ]
)