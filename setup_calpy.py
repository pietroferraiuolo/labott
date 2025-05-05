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
`calpy` is a command-line tool that calls an interactive Python 
shell (ipython3) with the option to pass the path to a configuration
file for the `aoptics` package.

Options:
--------
no option : Initialize an ipython3 --pylab='qt' shell

-f <path> : Option to pass the path to a configuration file to be read 
            (e.g., '../aopticsConf/configuration.yaml'). Used to initiate
            the aoptics package.

-f <path> --create : Create the configuration file in the specified path, 
                     as well as the complete folder tree. The `data_path`
                     variable in the created configuration file is autom-
                     atically set to the path of the configuration file.

-h |--help : Shows this help message

        """)
        sys.exit(0)
    elif len(sys.argv) > 2 and sys.argv[1] == '-f' and any([sys.argv[2] != '', sys.argv[2] != None]):
        config_path = sys.argv[2]
        if not any([config_path.startswith(home), config_path.startswith(mnt), config_path.startswith(media)]):
            config_path = os.path.join(home, config_path)
        if not '.yaml' in config_path:
            try:
                config_path = check_dir(config_path)
            except OSError as ose:
                print(f"Error: {ose}")
                sys.exit(1)
        if '--create' in sys.argv:
            from opticalib.core.root import create_configuration_file
            create_configuration_file(config_path, data_path=True)
        try:
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(config_path), 'SysConfig', 'configuration.yaml')
            print("\n Initiating IPython Shell, importing Aoptics...\n")
            os.system(f"export AOCONF={config_path} && ipython3 --pylab='qt' -i -c 'import aoptics'")
        except OSError as ose:
            print(f"Error: {ose}")
            sys.exit(1)
    elif len(sys.argv) == 1:
        os.system("ipython3 --pylab='qt'")
    else: # Handle invalid arguments
        print("Error: Invalid use. Use -h or --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()