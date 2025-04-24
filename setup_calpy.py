# -*- coding: utf-8 -*-
import os
import sys
import shutil

def check_dir(config_path):
    if not os.path.exists(config_path):
        os.makedirs(config_path)
        if not os.path.isdir(config_path):
            raise OSError(f"Invalid Path: {config_path}")
    config_path = os.path.join(config_path, 'configuration.yaml')
    return config_path

def main():
    home = os.path.expanduser("~")
    mnt = '/mnt/'
    media = '/media/'
    # Check if ipython3 is installed
    if not shutil.which("ipython3"):
        print("Error: ipython3 is not installed or not in your PATH.")
        sys.exit(1)
    # if -h/--help is passed, show help message
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
CALPY DOCUMENTATION
`calpy` is a command-line tool that calls an interactive Python shell (ipython3)
with the option to pass the path to a configuration file for the `aoptics` package.

Options:
    no option  : Initialize an ipython3 --pylab='qt' shell

    -f <path>  : Pass the path to a configuration file for the automatically
                imported `aoptics` package (e.g., '../aopticsConf/configuration.yaml')

    -h |--help : Shows this help message

        """)
        sys.exit(0)
    elif len(sys.argv) > 2 and sys.argv[1] == '-f':
        config_path = sys.argv[2]
        if not any([config_path.startswith(home), config_path.startswith(mnt), config_path.startswith(media)]):
            config_path = os.path.join(home, config_path)
        if not '.yaml' in config_path: # os.path.isfile(config_path):
            try:
                config_path = check_dir(config_path)
            except OSError as ose:
                print(f"Error: {ose}")
                sys.exit(1)
        if '--create' in sys.argv:
            from .aoptics.core.root import create_configuration_file
            create_configuration_file(config_path, data_path=True)
        if not os.path.exists(config_path):
            print(f"Error: The file {sys.argv[2]} does not exist.")
            sys.exit(1)
        os.system(f"export AOCONF={config_path} && ipython3 --pylab='qt' -i -c 'import aoptics'")
    else:
        os.system("ipython3 --pylab='qt'")

if __name__ == "__main__":
    main()