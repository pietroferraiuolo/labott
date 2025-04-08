'''
Author(s)
    -P.Ferraiuolo
    -R.Briguglio

Written in 06/2024; updated to use YAML on 04/2025
'''
import os as _os
import numpy as _np
import json as _json
import shutil as _sh
import yaml

from .root import (
    CONFIGURATION_FOLDER as _cfold,
    IFFUNCTIONS_ROOT_FOLDER as _iffold,
)

yaml_config_file = "configuration.yaml"

_nzeroName     = 'numberofzeros'
_modeIdName    = 'modeid'
_modeAmpName   = 'modeamp'
_templateName  = 'template'
_modalBaseName = 'modalbase'

_items = [_nzeroName, _modeIdName, _modeAmpName, _templateName, _modalBaseName]


def load_yaml_config(bpath=_cfold):
    """
    Loads the YAML configuration file.
    """
    fname = _os.path.join(bpath, yaml_config_file)
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    return config


def dump_yaml_config(config, bpath=_cfold):
    """
    Writes the configuration dictionary back to the YAML file.
    """
    fname = _os.path.join(bpath, yaml_config_file)
    with open(fname, 'w') as f:
        yaml.dump(config, f)


def getConfig(key, bpath=_cfold):
    """
    Reads the configuration from the YAML file for the IFF acquisition.
    The key passed is the block of information retrieved within the INFLUENCE.FUNCTIONS section.

    Parameters
    ----------
    key : str
        Key value of the block of information to read. Can be
            - 'TRIGGER'
            - 'REGISTRATION'
            - 'IFFUNC'
    bpath : str, optional
        Base path of the file to read. Default points to the configuration root folder.
            
    Returns
    -------
    info : dict
        A dictionary containing the configuration info:
            - zeros
            - modes
            - amplitude
            - template
            - modalBase
    """
    config = load_yaml_config(bpath)
    # The nested block is under INFLUENCE.FUNCTIONS
    cc = config["INFLUENCE.FUNCTIONS"][key]
    nzeros     = int(cc[_nzeroName])
    modeId     = _parse_val(cc[_modeIdName])
    modeAmp    = float(cc[_modeAmpName])
    modalBase  = cc[_modalBaseName]
    template   = _np.array(cc[_templateName])
    info = {
        'zeros': nzeros,
        'modes': modeId,
        'amplitude': modeAmp,
        'template': template,
        'modalBase': modalBase
    }
    return info


def copyConfigFile(tn, old_path=_cfold):
    """
    Copies the YAML configuration file to the new folder for record keeping of the 
    configuration used on data acquisition.

    Parameters
    ----------
    tn : str
        Tracking number for the new data.
    old_path : str, optional
        Base path where the YAML configuration file resides.

    Returns
    -------
    res : str
        Path where the file was copied.
    """
    config = load_yaml_config(old_path)
    nfname = _os.path.join(_iffold, tn, 'iffConfig.yaml')
    with open(nfname, 'w') as f:
        yaml.dump(config['INFLUENCE.FUNCTIONS'], f)
    print(f"IFF configuration copied to {nfname.rsplit('/' + yaml_config_file, 1)[0]}")
    return nfname


def updateConfigFile(key: str, item: str, value, bpath=_cfold):
    """
    Updates the YAML configuration file for the IFF acquisition.
    The key passed is within the INFLUENCE.FUNCTIONS section.

    Parameters
    ----------
    key : str
        Key of the block to update (e.g., 'TRIGGER', 'REGISTRATION', 'IFFUNC').
    item : str
        The configuration item to update.
    value : any
        New value to update.
    bpath : str, optional
        Base path of the configuration file.
    """
    if yaml_config_file not in bpath:
        fname = _os.path.join(bpath, yaml_config_file)
        # Create a backup if updating the master configuration
        if bpath == _cfold:
            fnameBck = _os.path.join(bpath, 'configuration_backup.yaml')
            _sh.copyfile(fname, fnameBck)
    else:
        fname = bpath
    config = load_yaml_config(bpath)
    if key not in config["INFLUENCE.FUNCTIONS"]:
        raise KeyError(f"Configuration section `{key}` not found in the YAML file")
    if item not in _items:
        raise KeyError(f"Item `{item}` not found in the configuration file")
    # Update the value (convert np.ndarray to list if needed)
    if isinstance(value, _np.ndarray):
        vmax = _np.max(value)
        vmin = _np.min(value)
        if _np.array_equal(value, _np.arange(vmin, vmax + 1)):
            config["INFLUENCE.FUNCTIONS"][key][item] = f"\"np.arange({vmin}, {vmax + 1})\""
        else:
            config["INFLUENCE.FUNCTIONS"][key][item] = str(value.tolist())
    else:
        config["INFLUENCE.FUNCTIONS"][key][item] = str(value)
    dump_yaml_config(config, bpath)


def getNActs_fromConf(bpath=_cfold):
    """
    Retrieves the number of actuators from the YAML configuration file.

    Parameters
    ----------
    bpath : str, optional
        Base path of the configuration file.

    Returns
    -------
    nacts : int
        Number of DM actuators.
    """
    config = load_yaml_config(bpath)
    dm_config = config["INFLUENCE.FUNCTIONS"]["DM"]
    nacts = int(dm_config['nacts'])
    return nacts


def getTiming(bpath=_cfold):
    """
    Retrieves timing information from the YAML configuration file.

    Parameters
    ----------
    bpath : str, optional
        Base path of the configuration file.

    Returns
    -------
    timing : int
        Timing used for synchronization.
    """
    config = load_yaml_config(bpath)
    dm_config = config["INFLUENCE.FUNCTIONS"]["DM"]
    timing = int(dm_config['timing'])
    return timing


def getCmdDelay(bpath=_cfold):
    """
    Retrieves the command delay from the YAML configuration file.

    Parameters
    ----------
    bpath : str, optional
        Base path of the configuration file.

    Returns
    -------
    cmdDelay : float
        Command delay for the interferometer synchronization.
    """
    config = load_yaml_config(bpath)
    dm_config = config["INFLUENCE.FUNCTIONS"]["DM"]
    cmdDelay = float(dm_config['delay'])
    return cmdDelay


def _parse_val(val):
    """
    Parses a value from the YAML configuration file.

    Parameters
    ----------
    val : str
        Value to parse.

    Returns
    -------
    parsed_val : int or float
        Parsed value, either as an integer or a float.
    """
    if isinstance(val, list):
        return _np.array(val)
    if isinstance(val, str):
        if val.startswith("np.arange"):
            return eval(val, {"np": _np})
        else:
            try:
                return eval(val)
            except Exception:
                return val
    return val