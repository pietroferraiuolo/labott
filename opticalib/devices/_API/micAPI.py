import os
import numpy as np
from abc import abstractmethod
try:
    from Microgate.adopt.AOClient import AO_CLIENT #type: ignore
except ImportError:
    pass


mirrorModesFile = 'ff_v_matrix.fits'
ffFile          = 'ff_matrix.fits'
actCoordFile    = 'ActuatorCoordinates.fits'
nActFile        = 'nActuators.dat'


class BaseAdOpticaDm():
    """
    Base class for AdOptica DM devices.
    This class is intended to be inherited by specific device classes.
    """
    def __init__(self, tracknum: str = None):
        """The constructor"""
        """
        print(f"Initializing the M4AU with configuration: '{os.path.join(fn.MIRROR_FOLDER,tracknum)}'")
        self.dmConf      = os.path.join(fn.MIRROR_FOLDER,tracknum)
        """
        self._aoClient = AO_CLIENT(tracknum)
        self.ffm = (self._aoClient.aoSystem.sysConf.gen.FFWDSvdMatrix)[0]# 
        self.ff  = self._aoClient.aoSystem.sysConf.gen.FFWDMatrix
        self._biasCmd    = self._aoClient.aoSystem.sysConf.gen.biasVectors[0]
        self.nActs       = self._initNActuators()
        self.mirrorModes = self._initMirrorModes()
        self.actCoord    = self._initActCoord()
        self.workingActs = self._initWorkingActs()
        self._aoClient.connect()

    def getCounter(self):
        """
        Function which returns the current shape of the mirror.

        Returns
        -------
        shape: numpy.ndarray
            Current shape of the mirror.
        """
        fc = self._aoClient.getCounters()
        skipByCommand = fc.skipByCommand
        #.....
        return skipByCommand

    def get_force(self):
        """
        Function which returns the current force applied to the mirror.

        Returns
        -------
        force: numpy.ndarray
            Current force applied to the mirror actuators.

        """
        #micLibrary.getForce()
        force = self._aoClient.getForce()
        return force


    def _initNActuators(self) -> int:
        """
        Function which reads the number of actuators of the DM from a configuration
        file.

        Returns
        -------
        nact: int
            number of actuators of the DM.
        """
        pass
        # fname = open(os.path.join(self.dmConf, nActFile),'r')
        # with open(fname,'r') as f:
        #     nact = int(f.read())
        # return nact


    def _initMirrorModes(self):
        """
        Function which initialize the mirror modes by reading from a fits file.

        Returns
        -------
        mirrorModes: numpy.ndarray
            Mirror Modes Matrix.
        """
        pass
        # fname = os.path.join(self.dmConf, mirrorModesFile)
        # if os.path.exists(fname):
        #     print('Initializing Mirror Modes')
        #     with pyfits.open(fname) as hdu:
        #         mirrorModes = hdu[0].data
        # else:
        #     print('Initializing Analytical Modal Base')
        #     mirrorModes = np.eye(self.nActs)
        # #nActs = np.shape(cmdMat)[0]
        # return mirrorModes


    def _initWorkingActs(self):
        """
        Function which initialize the working actuators by reading
        a list from a fits file.

        Returns
        -------
        workingActs: numpy.ndarray
            Working Actuators Matrix.
        """
        pass
        # fname = os.path.join(self.dmConf, mirrorModesFile)
        # if os.path.exists(fname):
        #     with pyfits.open(fname) as hdu:
        #         workingActs = hdu[0].data
        # else:
        #     workingActs = np.eye(self.nActs)
        # return workingActs
    

    def _initActCoord(self):
        '''
        Reading the actuators coordinate from file
        '''
        pass
        # fname = os.path.join(self.dmConf, actCoordFile)
        # with pyfits.open(fname) as hdu:
        #     actCoord = hdu[0].data
        # return actCoord
