"""
Author(s)
- Pietro Ferraiuolo : written in 2024

Description
-----------

"""
import os as _os
import numpy as _np
import time as _time

class AlpaoDm():
    """
    Alpao interface with M4 software.
    """

    def __init__(self, ip:str, port:int):
        """The Contructor"""
        import plico_dm
        print("Ricorda di spostare le _dmCoords!")
        self._dmCoords      = {
            'dm97' : [5, 7, 9, 11],
            'dm277': [7, 9, 11, 13, 15, 17, 19],
            'dm468': [8, 12, 16, 18, 20, 20, 22, 22, 24],
            'dm820': [10, 14, 18, 20, 22, 24, 26, 28, 28, 30, 30, 32],
        }
        self._dm            = plico_dm.deformableMirror(ip, port)
        self.nActs          = self._initNactuators()
        self.mirrorModes    = None
        self.actCoord       = self._initActCoord()
        self.cmdHistory     = None
        self.baseDataPath   = fn.OPD_IMAGES_ROOT_FOLDER
        self.refAct         = 425

    def get_shape(self):
        shape = self._dm.get_shape()
        return shape
    
    def set_shape(self, cmd, differential:bool=False):
        if differential:
            shape = self._dm.get_shape()
            cmd = cmd + shape
        self._checkCmdIntegrity(cmd)
        self._dm.set_shape(cmd)

    def uploadCmdHistory(self, cmdhist):
        self.cmdHistory = cmdhist

    def runCmdHistory(self, interf=None, delay=0.2, save:str=None, differential:bool=True):
        if self.cmdHistory is None:
            raise ValueError("No Command History to run!")
        else:
            tn = _ts.now() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = os.path.join(self.baseDataPath, tn)
            s = self.get_shape()
            if not os.path.exists(datafold) and interf is not None:
                os.mkdir(datafold)
            for i,cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd+s
                self.set_shape(cmd)
                if interf is not None:
                    time.sleep(delay)
                    img = interf.acquire_phasemap()
                    path = os.path.join(datafold, f"image_{i:05d}.fits")
                    rd.save_phasemap(path, img)
        self.set_shape(s)
        return tn

    def setZeros2Acts(self):
        zero = np.zeros(self.nActs)
        self.set_shape(zero)

    def nActuators(self):
        return self.nActs
    
    def _checkCmdIntegrity(self, cmd):
        mcmd = np.max(cmd)
        if mcmd > 0.9:
            raise ValueError(f"Command value {mcmd} is greater than 1.")
        mcmd = np.min(cmd)
        if mcmd < -0.9:
            raise ValueError(f"Command value {mcmd} is smaller than -1.")
        scmd = np.std(cmd)
        if scmd > 0.5:
            raise ValueError(f"Command standard deviation {scmd} is greater than 0.1.")

    def _initNactuators(self):
        return self._dm.get_number_of_actuators()

    def _initActCoord(self):
        nacts_row_sequence = self._dmCoords[f'dm{self.nActs}']
        n_dim = nacts_row_sequence[-1]
        upper_rows = nacts_row_sequence[:-1]
        lower_rows = [l for l in reversed(upper_rows)]
        center_rows = [n_dim]*upper_rows[0]
        rows_number_of_acts = upper_rows + center_rows + lower_rows
        N_acts = sum(rows_number_of_acts)
        n_rows = len(rows_number_of_acts)
        cx = np.array([], dtype=int)
        cy = np.array([], dtype=int)
        for i in range(n_rows):
            cx = np.concatenate((cx, np.arange(rows_number_of_acts[i]) + (n_dim - rows_number_of_acts[i]) // 2))
            cy = np.concatenate((cy, np.full(rows_number_of_acts[i], i)))
        self.actCoord = np.array([cx, cy])
        return self.actCoord
