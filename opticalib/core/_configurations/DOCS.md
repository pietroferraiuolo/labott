# CONFIGURATION FILE DOCUMENTATION

### Author(s)

- Pietro Ferraiuolo : written in 2025 (pietro.ferraiuolo@inaf.it)

### General description
The `configuration.yaml` is, as the name suggests, a configuration file for the optimal functioning of the `OptiCalib` python package, which, as of version `0.5.0`, includes four main sections:

- [System](#system-section)
- [Devices](#devices-section)
    - [Interferometer](#variables-overview-interferometers)
    - [Deformable Mirrors](#variables-overview-deformablemirrors)
- [Influence Functions](#iff-section)
- [Alignment](#alignment-section)

## System Section
### Description
In this first section of the configuration file, under the `SYSTEM` key, there are the general definitions the software needs to properly operate. Here we find the base data path, where obtained data and configurations are stored and retrieved, as well as various device configurations.

### Variables Overview
- **data_path : str**

    This is the base bath for the construction of the folder tree of the package.
    A copy of the `configuration.yaml` will be done inside the specified path, so that
    for future modifications there will be non need to touch the "root" configuration file.
    If left undefined, the package will automatically asign as base data path the home directory: 

    `~/opticalib_data/`

- **simulated.devices : bool**

    Here there is the indication on whether the devices are simulated or not.

    - **dm**

        Sets the deformable mirror device to simulated (`true`) or not (`false`).

    - **interf**

        Sets the interferometer device as simulated (`true`) or not (`false`).

## Devices Section
### Description
The devices part of the configuration file is divided into two main sections (soon to be 3?), one related to the Interferometers devices (under `INTERFEROMETERS`) and another for the Deformable Mirrors devices (under `DEFORMABLE.MIRRORS`). This section of the configuration serves to define key elements for the connection with the instruments, such as their `ip:port` adresses for ethernet connection.

### Variables Overview: `INTERFEROMETERS`
- **i4d_ip : str**

    This is the IP address of the interferometer instrument.

- **i4d_port : int**

    This is the network PORT of the interferometer.

- **Paths**

    In this sub-section are all the necessary paths to work with the interferometer.

    - **capture_4dpc : str**

        This is the `capture` folder in the proprietary `4D` computer, which needs to be network-mounted in the user machine.

    - **produce_4dpc : str**

        This is the `produce` folder in the proprietary `4D` computer, which needs to be network-mounted in the user machine.

    - **produce : str**

        This is the destination folder for the `produce`, which is in the user's machine.

    - **settings : str**

        This is the path to the `AppSettings.ini` configuration file of the interferometer, which needs to be network-mounted into the user's machine.


### Variables Overview: `DEFORMABLE.MIRRORS`

- **"device_name"** (Here goes the device name that has been defined, i.e. Splatt)

    - **ip : str**

        The `IP` address of the DM device to be connected.

    - **port : int**

        Network `PORT` of the DM device to be connected

<ins>NOTE</ins>: multiple devices can be defined in the configuration file, and actually it is recommended to define ahead all the possible available devices, to have a smoother experience on instancing the DM objects.


## IFF Section
### Description
In this section of the configuration file, under the `INFLUENCE.FUNCTIONs` key, are defined all the essential parameters needed to acquire the influence function for a deformable mirror.

### Variables Overview
There are four main parameter classes to define for the IFF acquisition:

#### `DM`
In here we find information related to the hardware behaviour of the deformable mirror:

- **nacts : int**

    The number of actuators of the deformable mirror in use for the acquisition

- **timing : int**

    The timing information is needed for the synchronization between the dm and the interferometer acquisition, ad is effectively done by repeating each column of the `timed command history` by the amount specified in this parameter. This is a tentative of replacement for the interferometer trigger line, which is the best way to perform synchronization between interferometer and Deformable Mirror's command actuation.

- **delay : float**

    This is the delay artificially introduced between an actuated command and it's subsequent interferometer acquisition, useful if the device at hand has a setting time for the commanded shape.

#### `TRIGGER - REGISTRATION - IFFUNC`
The following three parameter classes share the same identical structure, and define how is the `Timed Command Matrix History` (TCMH) os built, that is the command matrix that will be uploaded and run by the deformable mirror.

- `TRIGGER`

    The trigger, the first part of the `TCMH`, consists of a mode "shot" with high amplitude to identify the start of the measures on the resulting data cube.

- `REGISTRATION`

    The second part of the `TCMH` is the registration, where 3 (or more) actuators are individually actuated following a template, to allow the re-alignment of the images within the measured data, in the case of, for example, the system's position has drifted during the measurements.

- `IFFUNC`

    The final, bigger, part which makes up the `TCMH` is the Influence function itself, which is a series of modes repeated on a template, later analized to obtain the calibration.

As for the variables iside each parameter class, they are:

- **numberofzeros : int**

    This is the number of zeros at the start of each section. Serves as a sort of "spacing" between sections of the `TCMH`.

- **modeid : list or str**

    This variables indicates the modes to command. It can be an hand-written list or, more conveniently in the case of modes equal to the number of actuators of the DM, a string can be used to form an array in python, like

    ```yaml
    modeid : np.arange(0,88)
    ```

    In this way, `modeid` is the list of integers from 0 to 87, which tells the system to run all the modes from 1 to 88 on the selected modal base.

- **modeamp : float or list**

    This is the amplitude each command receives.<br> 
    If passed as a single float number, it is then transformed into an array with lenght equal to the number of measured modes, while if passed as list, it must have same size as the `modeid` array, as each value corresponds to the amplitude of the command at same inxed position.

- **template : list of int**

    A list of integers defining how each mode is executed. Usually is `[1,-1,1]`, which means that each mode is executed in a `push-pull-push` way.

- **modalbase : str**

    This string defines the modal base to use for the calibration. There are several options:

    - **zonal**

        This is the zonal modal base, which is represented by the identity matrix of dimentions $[N_{acts}\times N_{acts}]$ and means that each mode corresponds to a single actuator.

    - **hadamard**

        The [hadamard matrix](https://en.wikipedia.org/wiki/Hadamard_matrix), created as $H_{2^{10}}$ and then cut down on the number of actuators.

    - **mirror**

        This option uses the mirror's modes defined in the mirror class itself, which must be present as `*devicename*Mirror.fits` in the `ModalBases` folder.

    - **"custom"**

        A custom modal base is admitted, as long as the string put in the `modalbase` variable corresponds to the name of the actual modal base `.fits` file in the `ModalBases` folder.


## Alignment Section
### Description
In this section of the configuration files, are defined all the parameters and function calls of the various devices for the optical alignment of the system.

### <ins>Important Notes</ins>
The element order of the lists is <ins>**essential**</ins>, as the code will use the same index
for all the lists to access the right information. The order of the devices in the
lists must be the same as the order in the command matrix supplied, eg. if `device_y` 
is the second in place going through the command vector, then it's position must be
the second for every list in this configuration file.

### Variables Overview
- **names : list of str**
    
    This list contains the names of the devices, in the same order as all the 
    other lists. This is mainly used for logging purposes and print fancyness.<br>
    <ins>Use the same order of the devices used here for every subsequent variable in the configuration.</ins>

- **devices_move_calls : list of str**
    
    This list must contain the callables for moving each device, in the same 
    order as all the other lists.<br>

- **devices_read_calls : list of str**
    
    This list must contain the callables used to read the devices positions, in 
    the same order as all the other lists.

- **ccd_acquisition : list of str**
    
    This list contains the callables for the acquisition device (Interferometer, camera,...). The first item must be the acquisition method.

- **devices_dof : list of int | int**

    This variable represents the total degrees of freedom, at the software level, each device has. This parameter, then, will be the length of the accepted command vector by the devices. It can be an `int` (if all the devices use have the same number of accepted dof) or a list of integers, where each int corresponds to the dof of a device.

- **dof : list of lists**

    This variable, initialized as an empty list, will contain the degrees of
    freedom each device <ins>can actually move</ins>.<br>

    <ins>Example</ins>:<br>
    If a device can move only 3 DoF, say `piston`, `tip` and `tilt`, then, if the
    `devices_dof` for the device if 6 (i.e the device accept a vector of 6 elements), the 'dof' list should be the index at which these degrees of freedom are located in the
    accepted command vector. In this case, the list could be [2, 3, 4] (as 
    in the case for the M4's OTT) and the total vector would then be `[0,0,p,t,t,0]`.

- **slices : list**

    This variable is a list of slices that will be used to extract the right dof
    positions from the command matrix full vector.<br>
    Each slice is defined with the `start` and `stop` points:

    - **start : int**

        Starting point of the slice

    - **stop : int**

        Ending point of the slice (python notation, so the actual index used is `stop - 1`)

    <ins>Example</ins>:<br>
    If the command matrix is a column vector (the full command) of 7 elements,
    in which the first 3 elements are for the `device_1` dof, the second 2 
    for `device_2` dof and the last 3 for `device_3` dof, then the slices parameters should be:

    ```yaml
      slices:
        # device_1
        - start : 0
          stop  : 3
        # device_2
        - start : 3
          stop  : 5
        # device_3
        - start : 5
          stop  : 7
    ```

- **zernike_to_use : list of int**

    This variable contains the zernike modes indices which will be used for the 
    Interaction Matrix Construction, extracted from the general zernike fit. It is a list of integers, where each value corresponds to a mode order.

- **push_pull_template : list of int**

    Template for the push-pull algorithm. Being the command differential, the 
    push and pulls are to be written with respect to zero.<br>

    <ins>Example</ins>: A classic `[1,-1,1]` template, translates into `[1,-2,1]`.

- **commandMatrix : str**
    
    Control matrix file name, which must be a `.fits` file located under the `base_data_path/Alignment/ControlMatrices` folder.

- **fitting_surface : str**
    
    Path to the calibration fits file, whose mask will be used for zernike fitting
    purposes. If no mask is to be provided, leave it as an empty string.

### Builtin Example: M4's OTT
The template (root) configuration file has the alignment section filled with the configuration for the ELT@M4 Optical Test Tower (OTT). Here we we have three devices: the Parabola, the Reference Mirror and the M4 Exapode. Each device accept a vector of 6 elements, which means they "speak" in 6 DoF, however, for 'reality' constraints, they can only move certain dof. In particular: the Parabola can move in piston, tip and tilt, so the DoF are 2, 3 and 4; the Reference Mirror can move in tip and tilt, so the DoF are 3 and 4; the M4 Exapode can move in tip and tilt, so the DoF are 3 and 4. The command matrix (7x7) has a 7 element columns, which is the alignment command, and these devices appear in the same order as are listed above, so the ordering inside every list will be Parabola-ReferenceMirror-M4Exapode.