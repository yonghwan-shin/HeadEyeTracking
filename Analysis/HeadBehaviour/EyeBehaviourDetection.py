import numpy as np
from statsmodels.robust.scale import mad
from scipy import signal
from scipy import ndimage
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter
from math import (
    degrees,
    atan2,
)
import logging
lgr = logging.getLogger('remodnav.clf')

class EyeBehaviourClassifier():
    def __init__(self,data,
                 sampling_rate,
                 pursuit_velthresh=2.0,
                 noise_factor=5.0,
                 velthresh_startvelocity=300.0,
                 min_intersaccade_duration=0.04,
                 min_saccade_duration=0.01,
                 max_initial_saccade_freq=2.0,
                 saccade_context_window_length=1.0,
                 max_pso_duration=0.04,
                 min_fixation_duration=0.04,
                 min_pursuit_duration=0.04,
                 lowpass_cutoff_freq=4.0
                 ):
        self.data=data
        self.sr = sr = sampling_rate
        self.velthresh_startvel = velthresh_startvelocity
        self.lp_cutoff_freq = lowpass_cutoff_freq
        self.pursuit_velthresh = pursuit_velthresh
        self.noise_factor = noise_factor

        # convert to #samples
        self.min_intersac_dur = int(
            min_intersaccade_duration * sr)
        self.min_sac_dur = int(
            min_saccade_duration * sr)
        self.sac_context_winlen = int(
            saccade_context_window_length * sr)
        self.max_pso_dur = int(
            max_pso_duration * sr)
        self.min_fix_dur = int(
            min_fixation_duration * sr)
        self.min_purs_dur = int(
            min_pursuit_duration * sr)

        self.max_sac_freq = max_initial_saccade_freq / sr

    def _get_adaptive_saccade_velocity_velthresh(self,vels):
        cur_thresh = self.velthresh_startvel
        def _get_thresh(cut):
            vel_uthr = vels[vels < cut]
            med = np.median(vel_uthr)
            scale = mad(vel_uthr)
            return med + 2 * self.noise_factor * scale, med, scale

        # re-compute threshold until value converges
        count = 0
        dif = 2
        while dif > 1 and count < 30:  # less than 1deg/s difference
            old_thresh = cur_thresh
            cur_thresh, med, scale = _get_thresh(old_thresh)
            if not cur_thresh:
                # safe-guard in case threshold runs to zero in
                # case of really clean and sparse data
                cur_thresh = old_thresh
                break
            lgr.debug(
                'Saccade threshold velocity: %.1f '
                '(non-saccade mvel: %.1f, stdvel: %.1f)',
                cur_thresh, med, scale)
            dif = abs(old_thresh - cur_thresh)
            count += 1

        return cur_thresh, (med + self.noise_factor * scale)