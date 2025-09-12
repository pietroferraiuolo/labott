import os
import time
import numpy as np
from matplotlib import pyplot as plt
from opticalib import folders as fp
from opticalib.ground import osutils as osu, zernike as zern
from .base_fake_alpao import BaseFakeAlpao


class AlpaoDm(BaseFakeAlpao):

    def __init__(self, nActs: int):
        super(AlpaoDm, self).__init__(nActs)
        self.cmdHistory = None
        self._shape = np.ma.masked_array(self.mask * 0, mask=self.mask, dtype=float)
        self._idx = np.where(self.mask == 0)
        self._actPos = np.zeros(self.nActs)
        self._live = False
        self._produce_random_shape()

    def set_shape(self, command, differential: bool = False, modal: bool = False):
        """
        Applies the given command to the deformable mirror.

        Parameters
        ----------
        command : np.array
            Command to be applied to the deformable mirror.

        differential : bool
            If True, the command is applied differentially.
        """
        scaled_cmd = command * 1e-5  # more realistic command
        self._mirror_command(scaled_cmd, differential, modal)
        if self._live:
            import time

            time.sleep(0.15)
            plt.pause(0.05)

    def get_shape(self):
        """
        Returns the current amplitudes commanded to the dm's actuators.

        Returns
        -------
        np.array
            Current amplitudes commanded to the dm's actuators.
        """
        return self._actPos.copy()

    def uploadCmdHistory(self, cmdhist):
        """
        Upload the command history to the deformable mirror memory.
        Ready to run the `runCmdHistory` method.
        """
        self.cmdHistory = cmdhist

    def runCmdHistory(
        self,
        interf=None,
        save: str = None,
        rebin: int = 1,
        modal: bool = False,
        differential: bool = True,
        delay: float = 0,
    ):
        """
        Runs the command history on the deformable mirror.

        Parameters
        ----------
        interf : Interferometer
            Interferometer object to acquire the phase map.
        rebin : int
            Rebinning factor for the acquired phase map.
        modal : bool
            If True, the command history is modal.
        differential : bool
            If True, the command history is applied differentially
            to the initial shape.

        Returns
        -------
        tn :str
            Timestamp of the data saved.
        """
        if self.cmdHistory is None:
            raise Exception("No Command History to run!")
        else:
            tn = osu.newtn() if save is None else save
            print(f"{tn} - {self.cmdHistory.shape[-1]} images to go.")
            datafold = os.path.join(fp.OPD_IMAGES_FOLDER, tn)
            s = self.get_shape()
            if not os.path.exists(datafold):
                os.mkdir(datafold)
            for i, cmd in enumerate(self.cmdHistory.T):
                print(f"{i+1}/{self.cmdHistory.shape[-1]}", end="\r", flush=True)
                if differential:
                    cmd = cmd + s
                self.set_shape(cmd, modal=modal)
                if interf is not None:
                    time.sleep(delay)
                    img = interf.acquire_phasemap(rebin=rebin)
                    path = os.path.join(datafold, f"image_{i:05d}.fits")
                    osu.save_fits(path, img)
        self.set_shape(s)
        return tn

    def visualize_shape(self, cmd=None):
        """
        Visualizes the command amplitudes on the mirror's actuators.

        Parameters
        ----------
        cmd : np.array, optional
            Command to be visualized on the mirror's actuators. If none, will plot
            the current position of the actuators.

        Returns
        -------
        np.array
            Processed shape based on the command.
        """
        if cmd is None:
            cmd = self._actPos.copy()
        plt.figure(figsize=(7, 6))
        size = (120 * 97) / self.nActs
        plt.scatter(
            self._scaledActCoords[:, 0], self._scaledActCoords[:, 1], c=cmd, s=size
        )
        plt.xlabel(r"$x$ $[px]$")
        plt.ylabel(r"$y$ $[px]$")
        plt.title(f"DM {self.nActs} Actuator's Coordinates")
        plt.colorbar()
        plt.show()

    def _mirror_command(self, cmd, diff, modal):
        """
        Applies the given command to the deformable mirror.

        Parameters
        ----------
        cmd : np.array
            Command to be processed by the deformable mirror.

        diff : bool
            If True, process the command differentially.

        Returns
        -------
        np.array
            Processed shape based on the command.
        """
        if modal:
            mode_img = np.dot(self.ZM, cmd)
            cmd = np.dot(mode_img, self.RM)
        cmd_amp = cmd
        if not diff:
            cmd_amp = cmd - self._actPos
        self._shape[self._idx] += np.dot(cmd_amp, self.IM)
        self._actPos += cmd_amp

    def _wavefront(self, **kwargs):
        """
        Current shape of the mirror's surface. Only used for the interferometer's
        live viewer (see `interferometer.py`).

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for customization.
            - zernike : int ,
                Zernike mode to be removed from the wavefront.
            - surf : bool ,
                If True, the shape is returned instead of
                the wavefront.
            - noisy : bool ,
                If True, adds noise to the wavefront.

        Returns
        -------
        wf : np.array
            Phase map of the interferometer.
        """
        zernike = kwargs.get("zernike", None)
        surf = kwargs.get("surf", True)
        noisy = kwargs.get("noisy", False)
        img = np.ma.masked_array(self._shape, mask=self.mask)
        if zernike is not None:
            img = zern.removeZernike(img, zernike)
        if not surf:
            Ilambda = 632.8e-9
            phi = np.random.uniform(-0.25 * np.pi, 0.25 * np.pi) if noisy else 0
            wf = np.sin(2 * np.pi / Ilambda * img + phi)
            A = _geo.rms(img) / _geo.rms(wf)
            wf *= A
            img = wf
        return img

    def _produce_random_shape(self):
        """
        Produces a random shape for the deformable mirror initialization,
        by using a linear combination of Tip/Tilt and focus.

        Returns
        -------
        np.array
            Random shape for the deformable mirror.
        """
        try:
            shape = osu.load_fits(
                os.path.join(
                    fp.CONFIGURATION_ROOT_FOLDER, f"dm{self.nActs}_baseShape.fits"
                )
            )
            self._shape = np.ma.masked_array(shape)
        except FileNotFoundError:
            mat = np.eye(self.nActs)
            tx = mat[0]
            ty = mat[1]
            f = mat[3]
            rand = np.random.uniform
            cmd = (
                rand(0.05, 0.005) * ty
                + rand(0.05, 0.005) * tx
                + rand(0.005, 0.0005) * f
            )
            self.set_shape(cmd, modal=True)
            osu.save_fits(
                os.path.join(
                    fp.CONFIGURATION_ROOT_FOLDER, f"dm{self.nActs}_baseShape.fits"
                ),
                self._shape,
            )
            self._actPos = np.zeros(self.nActs)
