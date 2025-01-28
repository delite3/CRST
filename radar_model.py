import numpy as np
import numpy.typing as npt
from typing import Union


class RadarModel:
    # Helper class for simulation of Pd and SNR
    def __init__(
        self,
        snr_50_db: float = 3.768,
        rcs_50: float = 0.854,
        r_50: float = 70,
        pf: float = 1e-2,
        lobe_width_scale_factor: float = 0.2292,
        rcs_min: float = -30,
    ) -> None:
        # The default parameters seems to match the Bosch radar reasonably well

        # Determines the width of the simulated antenna lobe
        self.lobe_width_scale_factor = lobe_width_scale_factor

        # Parameters in linear scale
        self.snr_50 = 10 ** (snr_50_db / 10)
        self.rcs_50 = rcs_50
        self.r_50 = r_50

        self.pf = pf
        assert self.pf < 1

        # RCS drops to this value at 0m
        # Mimics the behavior of the Bosch radar
        self.rcs_min = rcs_min

    def compute_snr(
        self,
        r: Union[npt.NDArray[np.float64], float],
        phi: Union[npt.NDArray[np.float64], float],
        rcs: float,
    ) -> npt.NDArray[np.float64]:
        # Simulate and return the snr

        return (
            self.snr_50
            * (rcs / self.rcs_50)
            * ((self.r_50 / r) ** 4)
            * self.gaussian_gain(phi)
        )

    def gaussian_gain(
        self, phi: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.float64]:
        # Antenna lobe, approximated with a gaussian distribution.
        # Matches the figures in the Bosch data sheet reasonably well
        return np.exp(-0.5 * np.square(phi / self.lobe_width_scale_factor))

    def compute_pd(
        self, snr: Union[npt.NDArray[np.float64], float]
    ) -> npt.NDArray[np.float64]:
        # Simulate and return the Pd
        return self.pf ** (1 / (1 + snr))

    def sim_rcs(
        self,
        r: Union[npt.NDArray[np.float64], float],
        rcs: float,
        add_noise: bool = True,
    ) -> npt.NDArray[np.float64]:
        # Simulate the behavior of the Bosch radars RCS estimation
        # RCS is typically reasonably stable (but noisy) at larger distances, but starts
        # to drop sharply as the object gets closer to the radar.
        # It seems to correspond to object size, so the rcs value used for the simulation
        # is used to determine where the RCS should start falling.
        drop_at = rcs * 2
        slope = (rcs - self.rcs_min) / drop_at
        rcs_sim = slope * r + self.rcs_min
        rcs_sim = np.where(r > drop_at, rcs, rcs_sim)
        if add_noise:
            rcs_sim += r * (np.random.random(rcs_sim.shape) + 0.5) / 10
        return rcs_sim
