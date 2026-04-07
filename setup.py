from setuptools import find_packages, setup
from typing import List
def get_requirements():
    requirement_list : List[str] = [ ]
    try:
        with open('requirements.txt', 'r') as line:
            lib = line.readlines()
            
            for line in lib:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirement_list.append(requirement)
                    
    except FileNotFoundError as ferr:
        print(f"Setup Error : {ferr}")

    return requirement_list

print(get_requirements())

setup(
    name="cherrypick-ml",
    author="Sujal G Sanyasi",
    version="0.1.0",
    author_email="cherrypickml1@gmail.com",
    description="A lightweight ML orchestration library with preprocessing, anomaly detection, and explainability tools",
    long_description= open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    install_requires=get_requirements()
)
