import os
from setuptools import setup, find_packages
from setuptools.command.install import install

about = {}
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, 'aoptics', '__version__.py'), 'r') as _:
    exec(_.read(), about)

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# class CalpyFuncInstall(install):
#     """
#     Custom install command to write the calpy function in the `.bashrc`
#     or `.bash_functions` file.
#     """
#     def run(self):
#         install.run(self)
#         bashrc = os.path.expanduser("~/.bashrc")
#         bashf  = os.path.expanduser("~/.bash_functions")
#         source_check = """if [ -f ~/.bash_functions ]; then
#     . ~/.bash_functions
# fi"""
#         calpy = """
# # calpy function

# calpy() {
#     if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
#         cat << 'EOF'
# CALPY DOCUMENTATION
# `calpy` is a bash function that calls an interactive python shell (ipython3) with
# the option to pass the path to a configuration file for the `aoptics` package.

# Options
# -------
#         no option : Initialize an ipython3 --pylab='qt' shell

# '-f'    pass the path to a configuration file for the automatically imported
#         `aoptics` package (e.g. '../aopticsConf/configuration.yaml')

# '-h'    Shows this help message

# EOF

#     elif [ "$1" == "-f" ]; then
#         if [ -n "$2" ]; then
#             export AOCONF="$2"
#             ipython3 --pylab='qt' -i -c "import aoptics"
#         else
#             echo "Error: No configuration file provided. Use '-h' for help."
#         fi
#     else
#         ipython3 --pylab='qt'
#     fi
# }

# """
#         # Check if the function is already in the file
#         if not os.path.exists(bashrc):
#             with open(bashrc, 'w') as f:
#                 f.write("#!/bin/bash\n")
#         with open(bashrc, 'r') as f:
#             content = f.read()
#         if not source_check in content:
#             with open(bashrc, 'a') as f:
#                 f.write(f"\n{source_check}\n")
#         if not os.path.exists(bashf):
#             with open(bashf, 'w') as f:
#                 f.write("#!/bin/bash\n")
#         with open(bashf, 'r') as f:
#             content = f.read()
#         try:
#             if not calpy in content:
#                 with open(bashf, 'a') as f: 
#                     f.write(calpy)
#                 print("\nThe `calpy` function has been written to ~/.bash_functions.\n\
# Please source it with `source ~/.bashrc`.\nInstructions with `calpy -h`.")
#             else:
#                 print("\nThe `calpy` function is already in ~/.bash_functions.")
#         except Exception as e:
#             print(f"\nError writing the `calpy` function. {e}\n\
# Please add the calpy function manually.")



setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    maintainer=about["__maintainer__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url=about["__url__"],
    license=about["__license__"],
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    include_package_data=True,
    package_data={'aoptics': ['core/_configurations/*.yaml']},
    entry_points={
        'console_scripts': [
            'calpy=aoptics.__main__:main',
        ],
    },
)
