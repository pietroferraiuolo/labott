'''
Author(s)
    -P.Ferraiuolo
    -R.Briguglio
    
Written in 06/2024
'''
import os as _os
import numpy as _np
import json as _json
import shutil as _sh
import configparser as _cp
from .root import (
    CONFIGURATION_FOLDER as _cfold,
    IFFUNCTIONS_ROOT_FOLDER as _iffold,
)


_config          = _cp.ConfigParser()

iff_configFile   = 'iffConfig.ini'
_nzeroName       = 'numberofzeros'
_modeIdName      = 'modeid'
_modeAmpName     = 'modeamp'
_templateName    = 'template'
_modalBaseName   = 'modalbase'

_items = [_nzeroName, _modeIdName, _modeAmpName, _templateName, _modalBaseName]


def getConfig(key, bpath=_cfold):
    """
    Reads the configuration file for the IFF acquisition.
    The key passed is the block of information retrieved

    Parameters
    ----------
    key : str
        Key value of the block of information to read. Can be
            - 'TRIGGER'
            - 'REGISTRATION'
            - 'IFFUNC'
    bpath : str, OPTIONAL
        Base path of the file to read. Default points to the Configuration root
        folder
            
    Returns
    -------
    info : dict
        A dictionary containing all the configuration file's info:
            - nzeros
            - modeId
            - modeAmp 
            - template
            - modalBase 
    """
    np = _np
    fname = _os.path.join(bpath, iff_configFile)
    _config.read(fname)
    cc = _config[key]
    nzeros      = int(cc[_nzeroName])
    modeId_str  = cc[_modeIdName]
    try:
        modeId = np.array(_json.loads(modeId_str))
    except _json.JSONDecodeError:
        modeId = np.array(eval(modeId_str))
    modeAmp     = float(cc[_modeAmpName])
    modalBase   = cc[_modalBaseName]
    template    = np.array(_json.loads(cc[_templateName]))
    info = {'zeros': nzeros,
            'modes': modeId,
            'amplitude': modeAmp,
            'template': template,
            'modalBase': modalBase
        }
    return info


def copyConfingFile(tn, old_path=_cfold):
    """
    This function copies the configuration file to the new folder created for the
    IFF data, to keep record of the configuration used on data acquisition.

    Parameters
    ----------
    tn : str
        Tracking number of the new data.
    old_path : str, OPTIONAL
        Base path of the file to read. Default points to the Configuration root
        folder.

    Returns
    -------
    res : str
        String containing the path where the file has been copied
    """
    fname = _os.path.join(old_path, iff_configFile)
    nfname= _os.path.join(_iffold, tn, iff_configFile)
    res = _sh.copy2(fname, nfname)
    print(f"{iff_configFile} copied to {res.split('/iffConfig.ini')[0]}")
    return nfname


def updateConfigFile(key: str, item: str, value, bpath=_cfold):
    """
    Updates the configuration file for the IFF acquisition.
    The key passed is the block of information to update

    Parameters
    ----------
    key : str
        Key value of the block of information to update. Can be
            - 'TRIGGER'
            - 'REGISTRATION'
            - 'IFFUNC'
    item : str
        A dictionary containing all the configuration file's info:
            - nzeros
            - modeId
            - modeAmp 
            - template
            - modalBase 
    value : any
        Value to update in the configuration file.
    bpath : str, OPTIONAL
        Base path of the file to read. Default points to the Configuration root
        folder
    """
    if not iff_configFile in bpath:
        fname = _os.path.join(bpath, iff_configFile)
        # Create a backup of the original file if it is the one in the configuration root folder
        if bpath == _cfold:
            fnameBck = _os.path.join(bpath, 'iffConfig_backup.ini')
            _sh.copyfile(fname, fnameBck)
    else:
        fname = bpath
    content = getConfig(key, bpath)
    if not item in _items:
        raise KeyError(f"Item `{item}` not found in the configuration file")
    with open(fname, 'w') as configfile:
        _config[key][item] = str(value) if not isinstance(value, _np.ndarray) else value.tolist()
        _config.write(configfile)


def getNActs_fromConf(bpath=_cfold):
    """
    Retrieves the number of actuators from the iffConfig.ini file. 
    DEPRECATED

    Parameters
    ----------
    bpath : str, OPTIONAL
        Base path of the file to read. Default points to the Configuration root\
        folder

    Returns
    -------
    nacts : int
        Number of DM's used actuators

    """
    fname = _os.path.join(bpath, iff_configFile)
    _config.read(fname)
    cc = _config['DM']
    nacts = int(cc['NActs'])
    return nacts


def getTiming(bpath=_cfold):
    """
    Retrieves the timing information from the iffConfig.ini file
    DEPRECATED??

    Parameters
    ----------
    bpath : str, OPTIONAL
        Base path of the file to read. Default points to the Configuration root\
        folder

    Returns
    -------
    timing : int
        Timing for the synchronization with the mirrors working frequency
    """
    fname = _os.path.join(bpath, iff_configFile)
    _config.read(fname)
    cc = _config['DM']
    timing = int(cc['Timing'])
    return timing


def getCmdDelay(bpath=_cfold):
    """
    Retrieves the command delay information from the iffConfig.ini file.

    Parameters
    ----------
    bpath : str, OPTIONAL
        Base path of the file to read. Default points to the Configuration root\
        folder

    Returns
    -------
    cmdDelay : int
        Command delay for the synchronization with the interferometer.
    """
    fname = _os.path.join(bpath, iff_configFile)
    _config.read(fname)
    cc = _config['DM']
    cmdDelay = float(cc['delay'])
    return cmdDelay