import os
import sys
import shutil
import subprocess
import importlib.util

def check_dir(config_path: str) -> str:
    if not os.path.exists(config_path):
        os.makedirs(config_path)
        if not os.path.isdir(config_path):
            raise OSError(f"Invalid Path: {config_path}")
    config_path = os.path.join(config_path, 'configuration.yaml')
    return config_path

def backend_fallback() -> str:
    """Check if the preferred matplotlib backend is available, fallback if not.

    Parameters
    ----------
    preferred : str
        The preferred backend to use.

    Returns
    -------
    str
        The backend to use.
    """
    import matplotlib
    try:
        # Prefer Qt event loop integration; IPython accepts --pylab=qt
        matplotlib.use('qt', force=True)
        return 'qt'
    except Exception:
        return "auto"


def main():
    """Main function to handle command-line arguments and launch IPython shell with optional configuration.
    """
    home = os.path.expanduser("~")
    backend = backend_fallback()
    init_file = os.path.join(os.path.dirname(__file__), '__init_script__', 'initCalpy.py')
    # Check if IPython is installed in current interpreter
    if importlib.util.find_spec("IPython") is None:
        print("Error: IPython is not installed in this Python environment.")
        sys.exit(1)
    # if -h/--help is passed, show help message
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
CALPY DOCUMENTATION
`calpy` is a command-line tool that calls an interactive Python 
shell (IPython) with the option to pass the path to a configuration
file for the `opticalib` package.

Options:
--------
no option : Initialize an IPython --pylab=qt shell

-f <path> : Option to pass the path to a configuration file to be read 
            (e.g., '../opticalibConf/configuration.yaml'). Used to initiate
            the opticalib package.

-f <path> --create : Create the configuration file in the specified path, 
                     as well as the complete folder tree and enters the 
                     ipython session. The `data_path` variable in the 
                     created configuration file is automatically set to 
                     the path of the configuration file.

-c <path> : Create the configuration file in the specified path, as well as 
            the complete folder tree, and exit. The `data_path` variable in
            the created configuration file is automatically set to the path 
            of the configuration file.

-h |--help : Shows this help message

        """)
        sys.exit(0)
    elif len(sys.argv) > 2 and sys.argv[1] == '-f' and sys.argv[2]:
        config_path = sys.argv[2]
        # Use robust absolute path detection (works on Windows and Unix)
        if not os.path.isabs(config_path):
            config_path = os.path.join(home, config_path)
        if not '.yaml' in config_path:
            try:
                config_path = check_dir(config_path)
            except OSError as ose:
                print(f"Error: {ose}")
                sys.exit(1)
        if '--create' in sys.argv or '-c' in sys.argv:
            from opticalib.core.root import create_configuration_file
            create_configuration_file(config_path, data_path=True)
        try:
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(config_path), 'SysConfig', 'configuration.yaml')
            print("\n Initiating IPython Shell, importing Opticalib...\n")
            env = os.environ.copy()
            env["AOCONF"] = config_path
            # Launch IPython using the current interpreter for cross-platform compatibility
            args = [sys.executable, "-m", "IPython", f"--pylab={backend}", "-i", init_file]
            subprocess.run(args, env=env, check=False)
        except OSError as ose:
            print(f"Error: {ose}")
            sys.exit(1)
    elif len(sys.argv) > 2 and sys.argv[1] == '-c' and sys.argv[2]:
        config_path = sys.argv[2]
        # Use robust absolute path detection (works on Windows and Unix)
        if not os.path.isabs(config_path):
            config_path = os.path.join(home, config_path)
        if not '.yaml' in config_path:
            try:
                config_path = check_dir(config_path)
            except OSError as ose:
                print(f"Error: {ose}")
                sys.exit(1)
        from opticalib.core.root import create_configuration_file
        create_configuration_file(config_path, data_path=True)
        sys.exit(0)
    elif len(sys.argv) == 1:
        # Start plain IPython pylab session with Qt integration
        args = [sys.executable, "-m", "IPython", f"--pylab={backend}"]
        subprocess.run(args, check=False)
    else: # Handle invalid arguments
        print("Error: Invalid use. Use -h or --help for usage information.")
        sys.exit(1)


if __name__ == "__main__":
    main()