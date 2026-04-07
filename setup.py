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
    name="cherrypickml",
    author="Sujal G Sanyasi",
    version="0.1.0",
    author_email="cherrypickml1@gmail.com",
    license="MIT",
    description="A lightweight ML orchestration library with preprocessing, anomaly detection, and explainability tools",
    packages=find_packages(),
    install_requires=get_requirements()
)
