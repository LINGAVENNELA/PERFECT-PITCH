"""
YIN pitch detection algorithm implementation.

Reference:
A. de Cheveigné and H. Kawahara,
"YIN, a Fundamental Frequency Estimator for Speech and Music",
JASA, 2002.
"""

from __future__ import annotations

import numpy as np

from .detector import PitchDetector


class YinPitchDetector(PitchDetector):
    """
    Fundamental frequency estimator based on the YIN algorithm.
    """

    def __init__(
        self,
        threshold: float = 0.10,
        min_frequency: float = 80.0,
        max_frequency: float = 1000.0,
    ) -> None:

        self.threshold = threshold
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> float | None:

        signal = self._validate_audio(audio)

        min_tau, max_tau = self._compute_tau_bounds(
            sample_rate,
            signal.size,
        )

        difference = self._difference_function(
            signal,
            min_tau,
            max_tau,
        )

        cmndf = self._compute_cmndf(
            difference,
        )

        period = self._find_period(
            cmndf,
            min_tau,
        )

        if period is None:
            return None

        refined_period = self._refine_period(
            cmndf,
            period,
        )

        frequency = self._period_to_frequency(
            refined_period,
            sample_rate,
        )

        if not self._validate_frequency(frequency):
            return None

        return frequency

    def _validate_audio(
        self,
        audio: np.ndarray,
    ) -> np.ndarray:

        signal = np.asarray(
            audio,
            dtype=np.float32,
        )

        if signal.ndim != 1:
            raise ValueError(
                "YIN expects a one-dimensional audio frame."
            )

        if signal.size == 0:
            raise ValueError(
                "Audio frame is empty."
            )

        return signal

    def _compute_tau_bounds(
        self,
        sample_rate: int,
        frame_length: int,
    ) -> tuple[int, int]:

        min_tau = max(
            2,
            int(sample_rate / self.max_frequency),
        )

        max_tau = min(
            int(sample_rate / self.min_frequency),
            frame_length // 2,
        )

        if min_tau >= max_tau:
            raise ValueError(
                "Frame too short for configured frequency range."
            )

        return min_tau, max_tau
    
    def _difference_function(
        self,
        signal: np.ndarray,
        min_tau: int,
        max_tau: int,
    ) -> np.ndarray:
        """
        Compute the YIN difference function:

            d(τ) = Σ(x[j] − x[j + τ])²
        """

        difference = np.zeros(
            max_tau + 1,
            dtype=np.float32,
        )

        for tau in range(min_tau, max_tau + 1):

            delta = signal[:-tau] - signal[tau:]

            difference[tau] = np.dot(
                delta,
                delta,
            )

        return difference

    def _compute_cmndf(
        self,
        difference: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the Cumulative Mean Normalized Difference Function
        (CMNDF) from the YIN paper.
        """

        cmndf = np.ones_like(
            difference,
            dtype=np.float32,
        )

        running_sum = 0.0

        for tau in range(1, difference.size):

            running_sum += difference[tau]

            if running_sum > 0.0:

                cmndf[tau] = (
                    difference[tau]
                    * tau
                ) / running_sum

        return cmndf
    
    def _find_period(
        self,
        cmndf: np.ndarray,
        min_tau: int,
    ) -> int | None:
        """
        Find the first local minimum below the YIN threshold.
        """

        for tau in range(min_tau, cmndf.size - 1):

            if cmndf[tau] < self.threshold:

                while (
                    tau + 1 < cmndf.size
                    and cmndf[tau + 1] < cmndf[tau]
                ):
                    tau += 1

                return tau

        return None

    def _refine_period(
        self,
        cmndf: np.ndarray,
        period: int,
    ) -> float:
        """
        Refine the detected period using parabolic interpolation.
        """

        if period <= 0:
            return float(period)

        if period >= cmndf.size - 1:
            return float(period)

        left = cmndf[period - 1]
        center = cmndf[period]
        right = cmndf[period + 1]

        denominator = (
            left
            - (2.0 * center)
            + right
        )

        if np.isclose(denominator, 0.0):
            return float(period)

        offset = (
            0.5
            * (left - right)
            / denominator
        )

        return float(period) + offset
    
    def _period_to_frequency(
        self,
        period: float,
        sample_rate: int,
    ) -> float:

        if period <= 0.0:
            raise ValueError(
                "Period must be greater than zero."
            )

        return sample_rate / period

    def _validate_frequency(
        self,
        frequency: float,
    ) -> bool:

        if not np.isfinite(frequency):
            return False

        if frequency < self.min_frequency:
            return False

        if frequency > self.max_frequency:
            return False

        return True

    def _compute_confidence(
        self,
        cmndf: np.ndarray,
        period: float,
    ) -> float:
        """
        Internal confidence estimate.

        Confidence approaches 1.0 for strong periodic signals
        and approaches 0.0 for weak or noisy signals.
        """

        index = int(round(period))

        if index < 0 or index >= cmndf.size:
            return 0.0

        confidence = 1.0 - float(cmndf[index])

        return float(
            np.clip(
                confidence,
                0.0,
                1.0,
            )
        )

    def __repr__(
        self,
    ) -> str:

        return (
            f"{self.__class__.__name__}("
            f"threshold={self.threshold}, "
            f"min_frequency={self.min_frequency}, "
            f"max_frequency={self.max_frequency})"
        )