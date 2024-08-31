from setuptools import find_packages,setup


print("hello")

HYPEN_E_DOT = '-e .'

def get_requirements(file_path):
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name="ML-project-End",
    version='0.1',
    author="mayank",
    author_email="mailtomayankaggarwal@gmail.com",
    packages=find_packages(),
    requires=get_requirements('requirements.txt')

)

