"""
Author(s)
---------
- Pietro Ferraiuolo : written in 2025

Description
-----------

"""
import os as _os
import numpy as _np
import time as _time
from . import _API as _api
from opticalib import typings as _ot
from opticalib.core.root import OPD_IMAGES_ROOT_FOLDER as _opdi
from opticalib.ground.osutils import newtn as _ts, save_fits as _sf


class AlpaoDm(_api.BaseAlpaoMirror,_api.base_devices.BaseDeformableMirror):
    """
    Alpao Deformable Mirror interface.
    """

    def __init__(self, ip: str = None, port:int = None, nacts:int = None):
        """The Contructor"""
        super.__init__(ip, port, nacts)
        self.baseDataPath   = _opdi

    def get_shape(self) -> _ot.ArrayLike:
        shape = self._dm.get_shape()
        return shape
    
    def set_shape(self, cmd: _ot.ArrayLike, differential:bool=False) -> None:
        if differential:
            shape = self._dm.get_shape()
            cmd = cmd + shape
        self._checkCmdIntegrity(cmd)
        self._dm.set_shape(cmd)

    def setZeros2Acts(self):
        zero = _np.zeros(self.nActs)
        self.set_shape(zero)

    def uploadCmdHistory(self, cmdhist: _ot.MatrixLike) -> None:
        self.cmdHistory = cmdhist

    def runCmdHistory(self, interf: _ot.InterferometerDevice = None, delay: int | float = 0.2, save: str = None, differential: bool=True) -> str:
        if self.cmdHistory is None:
            raise ValueError("No Command History to run!")
        else:
            tn = _ts.now() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = _os.path.join(self.baseDataPath, tn)
            s = self.get_shape()
            if not _os.path.exists(datafold) and interf is not None:
                _os.mkdir(datafold)
            for i,cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd+s
                self.set_shape(cmd)
                if interf is not None:
                    _time.sleep(delay)
                    img = interf.acquire_phasemap()
                    path = _os.path.join(datafold, f"image_{i:05d}.fits")
                    _sf(path, img)
        self.set_shape(s)
        return tn


class SplattDm(_api.base_devices.BaseDeformableMirror):
    """
    SPLATT deformable mirror interface.
    """

    def __init__(self, ip: str = None, port: int = None):
        """The Constructor"""
        self._name          = 'Splatt'
        self._dm            = _api.SPLATTEngine(ip,port)
        self.nActs          = self._dm.nActs
        self.mirrorModes    = self._dm.mirrorModes
        self.actCoord       = self._dm.actCoords
        self.cmdHistory     = None
        self.baseDataPath   = _opdi
        self.refAct         = 16

    def get_shape(self):
        shape = self._dm.get_position()
        return shape

    def set_shape(self, cmd, differential:bool=False):
        if differential:
            lastCmd = self._dm.get_position_command()
            cmd = cmd + lastCmd
        self._checkCmdIntegrity(cmd)
        self._dm.set_position(cmd) 

    def uploadCmdHistory(self, cmdhist):
        self.cmdHistory = cmdhist

    def runCmdHistory(self, interf=None, delay=0.2, save:str=None, differential:bool=True):
        if self.cmdHistory is None:
            raise ValueError("No Command History to run!")
        else:
            tn = _ts.now() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = _os.path.join(self.baseDataPath, tn)
            s = self._dm.get_position_command()  #self._dm.flatPos # self.get_shape()
            if not _os.path.exists(datafold) and interf is not None:
                _os.mkdir(datafold)
            for i,cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd+s
                self.set_shape(cmd)
                if interf is not None:
                    _time.sleep(delay)
                    img = interf.acquire_map()
                    path = _os.path.join(datafold, f"image_{i:05d}.fits")
                    _sf(path, img)
        self.set_shape(s)
        return tn

    def sendBufferCommand(self, cmd, differential:bool=False, delay = 1.0):
        # cmd is a command relative to self._dm.flatPos
        if differential:
            lastCmd = self._dm.get_position_command()
            cmd = cmd + lastCmd
        self._checkCmdIntegrity(cmd) 
        cmd = cmd.tolist()
        tn = self._dm._eng.read(f'prepareCmdHistory({cmd})')
        #if accelerometers is not None:
        #   accelerometers.start_schedule()
        self._dm._eng.oneway_send(f'pause({delay}); sendCmdHistory(buffer)')
        return tn

    @property
    def nActuators(self):
        return self.nActs

    def _checkCmdIntegrity(self, cmd):
        pos = cmd + self._dm.flatPos
        if _np.max(pos) > 1.2e-3:
            raise ValueError(f'End position is too high at {_np.max(pos)*1e+3:1.2f} [mm]')
        if _np.min(pos) < 450e-6:
            raise ValueError(f'End position is too low at {_np.min(pos)*1e+3:1.2f} [mm]')