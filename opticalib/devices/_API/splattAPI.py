import Pyro4
import numpy as np
from opticalib.core.exceptions import DeviceNotFoundError


class SPLATTEngine:

    def __init__(self, ip: str, port: int):
        """The Constructor"""
        ip, port = self._get_address(ip, port)
        self._eng = Pyro4.Proxy(f"PYRO:matlab_engine@{ip}:{port}")
        self.nActs = int(self._eng.read("sys_data.mirrNAct"))
        self.actCoords = np.array(self._eng.read("mirrorData.coordAct"))
        self.mirrorModes = np.array(self._eng.read("sys_data.ff_v"))
        self.ffMatrix = np.array(self._eng.read("sys_data.ff_matrix"))
        self._bits2meters = float(self._eng.read("2^-sys_data.coeffs.Scale_F_Lin"))
        self._N2bits = float(self._eng.read("sys_data.coeffs.Force2DAC_V"))
        self._shellset = True
        try:
            self.flatPos = self.read_flat_data()
        except:
            self._shellset = False
            print(
                "Unable to read set position: remember to perform startup and set shell"
            )
        print("Initialized SPLATT deformable mirror")

    def get_position_command(self):  # relative to flatPos
        posCmdBits = np.array(self._eng.read("aoRead('sabu16_position',1:19)"))
        posCmd = posCmdBits * self._bits2meters
        posCmd = np.reshape(posCmd, self.nActs)
        posCmd -= self.flatPos
        return posCmd

    def get_position(self):
        pos = np.array(self._eng.read("lattGetPos()"))
        pos = np.reshape(pos, self.nActs)
        return pos

    def get_force(self):
        force = np.array(self._eng.read("lattGetForce()"))
        force = np.reshape(force, self.nActs)
        return force

    def set_position(self, cmd):
        if self._shellset is False:
            print("Shell must be set before giving commands!")
        cmd = cmd.tolist()
        self._eng.send(f"splattMirrorCommand({cmd}')")

    def read_buffers(
        self, external: bool = False, n_samples: int = 128, decimation: int = 0
    ):
        if n_samples > 256:
            raise ValueError("Maximum number of samples is 256!")
        self._eng.send(
            f"clear opts; opts.dec = {decimation}; opts.sampleNr = {n_samples}; opts.save2fits = 1; opts.save2mat = 0"
        )
        print("Reading buffers, hold tight: this may take a while ...")
        if external:
            self._eng.send(
                "[pos,cur,buf_tn]=splattAcqBufExt({'sabi32_Distance','sabi32_pidCoilOut'},opts)"
            )
        else:
            self._eng.send(
                "[pos,cur,buf_tn]=splattAcqBufInt({'sabi32_Distance','sabi32_pidCoilOut'},opts)"
            )
        buf_tn = self._eng.read("buf_tn")
        mean_pos = np.array(self._eng.read("mean(pos,2)")) * self._bits2meters
        mean_pos = np.reshape(mean_pos, self.nActs)
        mean_cur = np.array(self._eng.read("mean(cur,2)"))
        mean_cur = np.reshape(mean_cur, self.nActs)
        return mean_pos, mean_cur, buf_tn

    def saveFlatTN(self, tn: str = None):
        if tn is None:
            tn = self._eng.read("lattSaveFlat()")
        else:
            tn = self._eng.read(f"lattSaveFlat({tn})")
        return tn

    def updateFlatTN(self, tn: str = None):
        if tn is not None:
            self._eng.send(f"lattLoadFlat('{tn}')")
        self.flatPos = self.read_flat_data()

    def read_flat_data(self):
        flatPos = np.array(self._eng.read("sys_data.flatPos")) * self._bits2meters
        flatPos = np.reshape(flatPos, self.nActs)
        return flatPos

    def _set_shell(self):
        if self._shellset is False:
            print("Setting the shell...")
            self._eng.send("splattFastSet")
            self._shellset = True
        else:
            print("Shell set variable is True, overwrite it if you wish to set again")

    def _get_address(self, ip, port):
        from opticalib.core.read_config import getDmConfig

        try:
            config = getDmConfig("Splatt")
            rip, rport = config.get("ip"), config.get("port")
            if (ip, port) == (None, None):
                ip, port = (rip, rport)
        except KeyError:
            raise DeviceNotFoundError("Splatt")
        return ip, port
