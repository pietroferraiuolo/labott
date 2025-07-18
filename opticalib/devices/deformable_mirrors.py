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
from opticalib.core import exceptions as _oe
from opticalib.core.read_config import getDmIffConfig as _dmc


class AdOpticaDm(_api.BaseAdOpticaDm, _api.base_devices.BaseDeformableMirror):

    def __init__(self, tn: _ot.Optional[str] = None):
        """The Constructor"""
        self._name = "AdOpticaDm"
        super().__init__(tn)

    def get_shape(self):
        """
        Retrieve the actuators positions
        """
        pos = self._aoClient.getPosition()
        return pos

    def set_shape(self, cmd: list[float]): #cmd, segment=None):
        """
        Applies the given command to the DM actuators.

        Parameters
        ----------
        cmd : list[float]
            The command to be applied to the DM actuators, of lenght equal
            the number of actuators.
        """
        if not len(cmd) == self.nActs:
            raise _oe.CommandError(
                f"Command length {len(cmd)} does not match the number of actuators {self.nActs}."
            )
        self._aoClient.mirrorCommand(cmd)

    def uploadCmdHistory(self, tcmdhist: _ot.MatrixLike) -> None:
        """
        Uploads the (timed) command history in the DM. if `for_triggered` is true, 
        then it is loaded direclty in the AO client for the triggere mode run.

        Parameters
        ----------
        tcmdhist : _ot.MatrixLike
            The command history to be uploaded, of shape (used_acts, nmodes).
        tfor_triggered : bool, optional
            If True, the command history will be uploaded directly to the AO client for
            the triggered mode run. If False, it will be stored in the `cmdHistory`
            attribute of the DM instance (default is False).
        """
        if not _ot.isinstance_(tcmdhist, "MatrixLike"):
            raise _oe.MatrixError(
                f"Expecting a 2D Matrix of shape (used_acts, nmodes), got instead: {tcmdhist.shape}"
            )
        trig = _dmc()
        if trig is not False:
            self._tCmdHistory = tcmdhist.copy()
            self._aoClient.timeHistoryUpload(tcmdhist)
        else:
            self.cmdHistory = tcmdhist
        print("Time History uploaded!")

    def runCmdHistory(self, interf: _ot.Optional[_ot.InterferometerDevice] = None, differential: bool = False, save: _ot.Optional[str] = None) -> None:
        """
        Runs the loaded command history on the DM. If `triggered` is not False, it must
        be a dictionary containing the low lever arguments for the `aoClient.timeHistoryRun` function.

        Parameters
        ----------
        interf : _ot.InterferometerDevice
            The interferometer device to be used for acquiring images during the command history run.
        differential : bool, optional
            If True, the commands will be applied as differential commands (default is False).
        triggered : bool | dict[str, _ot.Any], optional 
            If False, the command history will be run in a sequential mode. 
            If not False, a dictionary must be provided, where it should contain the keys 
            'freq', 'wait', and 'delay' for the triggered mode.
        sequential_delay : int | float, optional
            The delay between each command execution in seconds (only if not in 
            triggered mode).
        save : str, optional
            If provided, the command history will be saved with this name as a timestamp.
        """
        dmifconf = _dmc()
        triggered = dmifconf["triggerMode"]
        sequential_delay = dmifconf["sequentialDelay"]
        if triggered is not False:
            for arg in triggered.keys():
                if not arg in ["frequency", "cmdDelay"]:
                    raise _oe.CommandError(
                        f"Invalid argument '{arg}' in triggered commands."
                    )
            freq = triggered.get("frequency", 1.0)
            tdelay = triggered.get("cmdDelay", 0.8)
            ins = _np.zeros(self.nActs)
            self._aoClient.timeHistoryRun(freq, 0, tdelay)
            nframes = self._tCmdHistory.shape[-1]
            if interf is not None:
                interf.capture(nframes-2, save)
            self.set_shape(ins)
        else:
            if self.cmdHistory is None:
                raise _oe.CommandError("No Command History to run!")
            else:
                tn = _ts() if save is None else save
                print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
                datafold = _os.path.join(self.baseDataPath, tn)
                s = self.get_shape() - self._biasCmd
                if not _os.path.exists(datafold) and interf is not None:
                    _os.mkdir(datafold)
                for i, cmd in enumerate(self.cmdHistory.T):
                    print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                    if differential:
                        cmd = cmd + s
                    self.set_shape(cmd)
                    if interf is not None:
                        _time.sleep(sequential_delay)
                        img = interf.acquire_map()
                        path = _os.path.join(datafold, f"image_{i:05d}.fits")
                        _sf(path, img)


class AlpaoDm(_api.BaseAlpaoMirror, _api.base_devices.BaseDeformableMirror):
    """
    Alpao Deformable Mirror interface.
    """

    def __init__(
        self,
        nacts: _ot.Optional[int | str] = None,
        ip: _ot.Optional[str] = None,
        port: _ot.Optional[int] = None,
    ):
        """The Contructor"""
        super().__init__(ip, port, nacts)
        self.baseDataPath = _opdi

    def get_shape(self) -> _ot.ArrayLike:
        shape = self._dm.get_shape()
        return shape

    def set_shape(self, cmd: _ot.ArrayLike, differential: bool = False) -> None:
        if differential:
            shape = self._dm.get_shape()
            cmd = cmd + shape
        self._checkCmdIntegrity(cmd)
        self._dm.set_shape(cmd)

    def setZeros2Acts(self):
        zero = _np.zeros(self.nActs)
        self.set_shape(zero)

    def uploadCmdHistory(self, tcmdhist: _ot.MatrixLike) -> None:
        if not _ot.isinstance_(tcmdhist, "MatrixLike"):
            raise _oe.MatrixError(
                f"Expecting a 2D Matrix of shape (used_acts, nmodes), got instead: {tcmdhist.shape}"
            )
        self.cmdHistory = tcmdhist

    def runCmdHistory(
        self,
        interf: _ot.InterferometerDevice = None,
        delay: int | float = 0.2,
        save: str = None,
        differential: bool = True,
    ) -> str:
        if self.cmdHistory is None:
            raise _oe.MatrixError("No Command History to run!")
        else:
            tn = _ts.now() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = _os.path.join(self.baseDataPath, tn)
            s = self.get_shape()
            if not _os.path.exists(datafold) and interf is not None:
                _os.mkdir(datafold)
            for i, cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd + s
                self.set_shape(cmd)
                if interf is not None:
                    _time.sleep(delay)
                    img = interf.acquire_map()
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
        self._name = "Splatt"
        self._dm = _api.SPLATTEngine(ip, port)
        self.nActs = self._dm.nActs
        self.mirrorModes = self._dm.mirrorModes
        self.actCoord = self._dm.actCoords
        self.cmdHistory = None
        self.baseDataPath = _opdi
        self.refAct = 16

    def get_shape(self):
        shape = self._dm.get_position()
        return shape

    def set_shape(self, cmd: _ot.ArrayLike, differential: bool = False) -> None:
        if differential:
            lastCmd = self._dm.get_position_command()
            cmd = cmd + lastCmd
        self._checkCmdIntegrity(cmd)
        self._dm.set_position(cmd)

    def uploadCmdHistory(self, tcmdhist: _ot.MatrixLike) -> None:
        if not _ot.isinstance_(tcmdhist, "MatrixLike"):
            raise _oe.MatrixError(
                f"Expecting a 2D Matrix of shape (used_acts, nmodes), got instead: {tcmdhist.shape}"
            )
        self.cmdHistory = tcmdhist

    def runCmdHistory(
        self,
        interf: _ot.Optional[_ot.InterferometerDevice] = None,
        delay: int | float = 0.2,
        save: _ot.Optional[str] = None,
        differential: bool = True,
    ) -> str:
        if self.cmdHistory is None:
            raise _oe.MatrixError("No Command History to run!")
        else:
            tn = _ts() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = _os.path.join(self.baseDataPath, tn)
            s = self._dm.get_position_command()  # self._dm.flatPos # self.get_shape()
            if not _os.path.exists(datafold) and interf is not None:
                _os.mkdir(datafold)
            for i, cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd + s
                self.set_shape(cmd)
                if interf is not None:
                    _time.sleep(delay)
                    img = interf.acquire_map()
                    path = _os.path.join(datafold, f"image_{i:05d}.fits")
                    _sf(path, img)
        self.set_shape(s)
        return tn

    def sendBufferCommand(
        self, cmd: _ot.ArrayLike, differential: bool = False, delay: int | float = 1.0
    ) -> str:
        # cmd is a command relative to self._dm.flatPos
        if differential:
            lastCmd = self._dm.get_position_command()
            cmd = cmd + lastCmd
        self._checkCmdIntegrity(cmd)
        cmd = cmd.tolist()
        tn = self._dm._eng.read(f"prepareCmdHistory({cmd})")
        # if accelerometers is not None:
        #   accelerometers.start_schedule()
        self._dm._eng.oneway_send(f"pause({delay}); sendCmdHistory(buffer)")
        return tn

    @property
    def nActuators(self) -> int:
        return self.nActs

    def _checkCmdIntegrity(self, cmd: _ot.ArrayLike) -> None:
        pos = cmd + self._dm.flatPos
        if _np.max(pos) > 1.2e-3:
            raise _oe.CommandError(
                f"End position is too high at {_np.max(pos)*1e+3:1.2f} [mm]"
            )
        if _np.min(pos) < 450e-6:
            raise _oe.CommandError(
                f"End position is too low at {_np.min(pos)*1e+3:1.2f} [mm]"
            )
