from copy import deepcopy


BCI_CHANNELS = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1',
                'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
                'P1', 'Pz', 'P2', 'POz']
SLEEPPHYSIONET_CHANNELS = ['Fpz', 'Pz']


BCI_SFREQ = 250
SLEEPPHYSIONET_SFREQ = 100


DEFAULT_AUG_PARAMS_BCI = {
    'GaussianNoise': {
        'param_key': 'std',
        'range': (0., 0.2),
    },
    'FrequencyShift': {
        'param_key': 'max_delta_freq',
        'range': (0., 3.),
        'sfreq': BCI_SFREQ
    },
    'FTSurrogate': {
        'param_key': 'phase_noise_magnitude',
        'range': (0., 1.),
        'channel_indep': False,
    },
    'SmoothTimeMask': {
        'param_key': 'mask_len_samples',
        'range': (0, 500)
    },
    'ChannelsDropout': {
        'param_key': 'p_drop',
        'range': (0., 1.)
    },
    'ChannelsShuffle': {
        'param_key': 'p_shuffle',
        'range': (0., 1.)
    },
    'IdentityTransform': {
        'param_key': None,
        'range': None
    },
    'BandstopFilter': {
        'param_key': 'bandwidth',
        'range': (0., 2.),
        'sfreq': BCI_SFREQ,
        'max_freq': 30,
    },
    'SensorsXRotation': {
        'param_key': 'max_degrees',
        'range': (0., 30.),
        'ordered_ch_names': BCI_CHANNELS
    },
    'SensorsYRotation': {
        'param_key': 'max_degrees',
        'range': (0., 30.),
        'ordered_ch_names': BCI_CHANNELS
    },
    'SensorsZRotation': {
        'param_key': 'max_degrees',
        'range': (0., 30.),
        'ordered_ch_names': BCI_CHANNELS
    },
}


DEFAULT_AUG_PARAMS_SLEEP = deepcopy(DEFAULT_AUG_PARAMS_BCI)

DEFAULT_AUG_PARAMS_SLEEP[
    "SensorsXRotation"]["ordered_ch_names"] = SLEEPPHYSIONET_CHANNELS
DEFAULT_AUG_PARAMS_SLEEP[
    "SensorsYRotation"]["ordered_ch_names"] = SLEEPPHYSIONET_CHANNELS
DEFAULT_AUG_PARAMS_SLEEP[
    "SensorsZRotation"]["ordered_ch_names"] = SLEEPPHYSIONET_CHANNELS
DEFAULT_AUG_PARAMS_SLEEP[
    "FrequencyShift"]["sfreq"] = SLEEPPHYSIONET_SFREQ
DEFAULT_AUG_PARAMS_SLEEP[
    "BandstopFilter"]["sfreq"] = SLEEPPHYSIONET_SFREQ
DEFAULT_AUG_PARAMS_SLEEP[
    "SmoothTimeMask"]["range"] = (0, 200)
DEFAULT_AUG_PARAMS_SLEEP[
    "FTSurrogate"]["channel_indep"] = True

DEFAULT_AUG_PARAMS = {
    "SleepPhysionet": DEFAULT_AUG_PARAMS_SLEEP,
    "BCI": DEFAULT_AUG_PARAMS_BCI,
    "BCI_CROP": DEFAULT_AUG_PARAMS_BCI,
}


def get_augmentation(
        Augmentation,
        magnitude=None,
        param_key=None,
        range=None,
        probability=0.5,
        **kwargs):
    """Creates an instance of a braindecode.augmentation class with adjusted
    parameters.

    Parameters
    ----------
    Augmentation: braindecode.augmentation class
        The augmentation to instantiate
    magnitude: float
        The magnitude of the augmentation. A strength in the interval [0, 1]
        that will modulate the effects of the transformation.
    param_key: str
        The name of the parameter to modulate.
    range: tuple, tuple of tuples
        The interval (min, max) which sets the boundaries of possible
        parameter values. Beware that the type of min and max values
        is important. Indeed the chosen parameter will have the same
        type.
    probability: float
        The probability to apply the transformation.
    kwargs:
        Keywords args for the augmentation.

    Returns
    -------
    braindecode.augmentation instance

    Note
    ----
        If param_key is not parsed, the adjusted parameter will be the
        probability to apply the transformation.
    """
    if not param_key:
        return Augmentation(probability, **kwargs)

    else:
        assert isinstance(range[0], (int, float))
        param_type = type(range[0])
        param_val = magnitude * range[1] + (1 - magnitude) * range[0]
        param_val = param_type(param_val)

        return Augmentation(probability, **{param_key: param_val}, **kwargs)
