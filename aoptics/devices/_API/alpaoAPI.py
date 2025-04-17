import numpy as _np
from aoptics.core.read_config import getDmAddress

class BaseAlpaoMirror():

    def __init__(self, ip: str, port: int, nActs: int):
        import plico_dm
        self._dmCoords      = {
            'dm97' : [5, 7, 9, 11],
            'dm277': [7, 9, 11, 13, 15, 17, 19],
            'dm468': [8, 12, 16, 18, 20, 20, 22, 22, 24],
            'dm820': [10, 14, 18, 20, 22, 24, 26, 28, 28, 30, 30, 32],
        }
        self._dm            = self._init_dm(ip, port, nActs)
        self.nActs          = self._initNactuators()
        self._name          = f"Alpao{self.nActs}"
        self.actCoord       = self._initActCoord()
        self.mirrorModes    = None
        self.cmdHistory     = None
        self.refAct         = None

    @property
    def nActuators(self):
        return self.nActs
    
    def setReferenceActuator(self, refAct:int):
        if refAct < 0 or refAct > self.nActs:
            raise ValueError(f"Reference actuator {refAct} is out of range.")
        self.refAct = refAct
    
    def _checkCmdIntegrity(self, cmd):
        mcmd = _np.max(cmd)
        if mcmd > 0.9:
            raise ValueError(f"Command value {mcmd} is greater than 1.")
        mcmd = _np.min(cmd)
        if mcmd < -0.9:
            raise ValueError(f"Command value {mcmd} is smaller than -1.")
        scmd = _np.std(cmd)
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
        cx = _np.array([], dtype=int)
        cy = _np.array([], dtype=int)
        for i in range(n_rows):
            cx = _np.concatenate((cx, _np.arange(rows_number_of_acts[i]) + (n_dim - rows_number_of_acts[i]) // 2))
            cy = _np.concatenate((cy, _np.full(rows_number_of_acts[i], i)))
        self.actCoord = _np.array([cx, cy])
        return self.actCoord
    
    def _init_dm(self, ip, port, nacts):
        import plico_dm
        if (ip, port) == (None, None) and nacts is not None:
            name = f"Alpao{nacts}"
            self.ip, self.port = getDmAddress(name)
        elif (ip, port, nacts) == (None, None, None):
            raise ValueError("Either (ip, port) or nacts must be provided.")
        else:
            self.ip, self.port = ip, port
        return plico_dm.deformableMirror(self.ip, self.port)