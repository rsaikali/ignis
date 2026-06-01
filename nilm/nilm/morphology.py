"""
Morphological analysis module for appliance power signatures.

This module provides advanced feature extraction for power consumption patterns,
going beyond simple duration and average power to capture:
- Shape patterns (ramp, oscillating, multi-phase, etc.)
- Statistical moments (skewness, kurtosis)
- Gradient features (rise/fall rates)
- Plateau detection
- Oscillation characteristics
- Frequency domain analysis
"""

import logging
from datetime import datetime

import numpy as np
from scipy import signal, stats
from scipy.fft import fft, fftfreq


logger = logging.getLogger(__name__)


class MorphologyAnalyzer:
    """Analyzer for extracting morphological features from power signatures."""

    def __init__(self, sampling_rate_hz=1.0):
        """
        Initialize the morphology analyzer.

        Args:
            sampling_rate_hz: Sampling rate of the power data (Hz)
        """
        self.sampling_rate = sampling_rate_hz

    def analyze(self, power_values, start_time):
        """
        Perform complete morphological analysis on power signature.

        Args:
            power_values: Array of power values (W)
            start_time: Start timestamp of the signature

        Returns:
            Dictionary with morphological analysis results
        """
        if len(power_values) < 10:
            logger.warning(f"Not enough data points: {len(power_values)}")
            return self._empty_analysis()

        try:
            analysis = {
                "version": "1.0",
                "computed_at": datetime.utcnow().isoformat(),
                "basic_stats": self._compute_basic_stats(power_values),
                "shape_features": self._compute_shape_features(power_values),
                "gradient": self._compute_gradient_features(power_values),
                "plateaus": self._detect_plateaus(power_values),
                "oscillation_features": self._compute_oscillation_features(power_values),
                "statistical_moments": self._compute_statistical_moments(power_values),
                "frequency_domain": self._compute_frequency_features(power_values),
            }

            # Add phases detection if multi-phase pattern
            if analysis["shape_features"]["pattern_type"] == "multi_phase":
                analysis["phases"] = self._detect_phases(power_values)

            return analysis

        except Exception as e:
            logger.error(f"Error during morphological analysis: {e}")
            return self._empty_analysis()

    def _empty_analysis(self):
        """Return empty analysis structure."""
        return {
            "version": "1.0",
            "computed_at": datetime.utcnow().isoformat(),
            "error": "Insufficient data for analysis",
        }

    def _compute_basic_stats(self, power):
        """Compute basic statistical features."""
        duration_sec = len(power) / self.sampling_rate

        return {
            "duration_sec": float(duration_sec),
            "num_points": int(len(power)),
            "mean_power": float(np.mean(power)),
            "std_power": float(np.std(power)),
            "min_power": float(np.min(power)),
            "max_power": float(np.max(power)),
            "total_energy_wh": float(np.sum(power) / 3600.0),
        }

    def _compute_shape_features(self, power):
        """Compute shape-related features."""
        # Normalize power for shape analysis
        power_normalized = (power - np.min(power)) / (np.max(power) - np.min(power) + 1e-7)

        # Count transitions (changes > 10% of range)
        threshold = 0.1
        diff = np.abs(np.diff(power_normalized))
        num_transitions = np.sum(diff > threshold)

        # Compute smoothness (inverse of variation coefficient)
        variation_coef = np.std(power) / (np.mean(power) + 1e-7)
        smoothness = 1.0 / (1.0 + variation_coef)

        # Determine pattern type
        pattern_type = self._classify_pattern(power_normalized)

        return {
            "pattern_type": pattern_type,
            "num_transitions": int(num_transitions),
            "smoothness": float(smoothness),
            "variation_coefficient": float(variation_coef),
        }

    def _classify_pattern(self, power_norm):
        """Classify the power pattern type."""
        # Calculate gradient
        gradient = np.gradient(power_norm)

        # Check if mostly increasing (ramp up)
        if np.mean(gradient[gradient > 0]) > 0.01 and np.sum(gradient > 0) > len(gradient) * 0.7:
            return "ramp_up"

        # Check if mostly decreasing (ramp down)
        if np.mean(gradient[gradient < 0]) < -0.01 and np.sum(gradient < 0) > len(gradient) * 0.7:
            return "ramp_down"

        # Check for oscillations
        std_power = np.std(power_norm)
        if std_power > 0.15:
            # Find peaks
            peaks, _ = signal.find_peaks(power_norm, prominence=0.1)
            if len(peaks) > 3:
                return "oscillating"

        # Check for multiple phases
        # Use simple thresholding to find different levels
        hist, bin_edges = np.histogram(power_norm, bins=10)
        if np.sum(hist > len(power_norm) * 0.1) > 2:
            return "multi_phase"

        # Default: constant
        return "constant"

    def _compute_gradient_features(self, power):
        """Compute gradient-based features."""
        gradient = np.gradient(power) * self.sampling_rate

        return {
            "max_rise_rate": float(np.max(gradient)),
            "max_fall_rate": float(np.min(gradient)),
            "avg_abs_gradient": float(np.mean(np.abs(gradient))),
            "gradient_std": float(np.std(gradient)),
        }

    def _detect_plateaus(self, power):
        """Detect stable power plateaus."""
        plateaus = []

        # Use rolling window to find stable regions
        window_size = max(10, int(len(power) * 0.05))

        i = 0
        while i < len(power) - window_size:
            window = power[i : i + window_size]
            std = np.std(window)
            mean = np.mean(window)

            # If stable (low std relative to mean)
            if std < mean * 0.05:
                # Extend plateau as long as it stays stable
                end = i + window_size
                while end < len(power):
                    if np.abs(power[end] - mean) < std * 2:
                        end += 1
                    else:
                        break

                duration = (end - i) / self.sampling_rate

                # Only keep significant plateaus (> 10 seconds)
                if duration > 10:
                    plateaus.append(
                        {
                            "start_sec": float(i / self.sampling_rate),
                            "end_sec": float(end / self.sampling_rate),
                            "duration_sec": float(duration),
                            "avg_power": float(mean),
                            "std": float(std),
                        }
                    )

                i = end
            else:
                i += window_size // 2

        return plateaus

    def _compute_oscillation_features(self, power):
        """Compute oscillation characteristics."""
        # Find peaks
        peaks, properties = signal.find_peaks(power, prominence=np.std(power) * 0.5, distance=5)

        is_oscillating = len(peaks) > 2

        if not is_oscillating:
            return {"is_oscillating": False, "num_peaks": 0}

        # Calculate peak regularity (inverse of std of intervals)
        if len(peaks) > 1:
            intervals = np.diff(peaks) / self.sampling_rate
            regularity = 1.0 / (1.0 + np.std(intervals))
            avg_freq = 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0
        else:
            regularity = 0.0
            avg_freq = 0.0

        return {
            "is_oscillating": True,
            "num_peaks": int(len(peaks)),
            "oscillation_frequency_hz": float(avg_freq),
            "peak_regularity": float(regularity),
            "avg_peak_amplitude": float(np.mean(properties.get("prominences", [0]))),
        }

    def _compute_statistical_moments(self, power):
        """Compute higher-order statistical moments."""
        return {
            "skewness": float(stats.skew(power)),
            "kurtosis": float(stats.kurtosis(power)),
            "variance": float(np.var(power)),
        }

    def _compute_frequency_features(self, power):
        """Compute frequency domain features using FFT."""
        # Apply FFT
        n = len(power)
        if n < 20:
            return {"error": "Too few points for FFT"}

        # Detrend signal
        power_detrended = signal.detrend(power)

        # Apply window to reduce spectral leakage
        window = signal.windows.hann(n)
        power_windowed = power_detrended * window

        # Compute FFT
        fft_vals = fft(power_windowed)
        freqs = fftfreq(n, 1.0 / self.sampling_rate)

        # Only positive frequencies
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        fft_mag = np.abs(fft_vals[pos_mask])

        # Find dominant frequencies (top 3 peaks)
        peaks, _ = signal.find_peaks(fft_mag, prominence=np.max(fft_mag) * 0.1)

        if len(peaks) > 0:
            # Sort by magnitude
            sorted_indices = np.argsort(fft_mag[peaks])[::-1]
            dominant_freqs = freqs_pos[peaks[sorted_indices[:3]]].tolist()
        else:
            dominant_freqs = []

        # Spectral entropy
        psd = fft_mag**2
        psd_norm = psd / (np.sum(psd) + 1e-10)
        spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10)) / np.log2(len(psd_norm))

        return {
            "dominant_frequencies": [float(f) for f in dominant_freqs],
            "spectral_entropy": float(spectral_entropy),
            "has_harmonics": len(dominant_freqs) > 1,
        }

    def _detect_phases(self, power):
        """Detect distinct phases in multi-phase patterns."""
        # Use change point detection
        gradient = np.abs(np.gradient(power))

        # Smooth gradient
        window_size = max(5, int(len(gradient) * 0.02))
        gradient_smooth = np.convolve(gradient, np.ones(window_size) / window_size, mode="same")

        # Find change points (peaks in gradient)
        threshold = np.mean(gradient_smooth) + np.std(gradient_smooth)
        change_points = np.where(gradient_smooth > threshold)[0]

        if len(change_points) == 0:
            return []

        # Merge close change points
        merged_points = [change_points[0]]
        min_distance = int(30 * self.sampling_rate)  # 30 seconds minimum

        for cp in change_points[1:]:
            if cp - merged_points[-1] > min_distance:
                merged_points.append(cp)

        # Create phases
        phases = []
        merged_points = [0] + merged_points + [len(power)]

        for i in range(len(merged_points) - 1):
            start = merged_points[i]
            end = merged_points[i + 1]

            if end - start < 10:  # Skip very short phases
                continue

            phase_power = power[start:end]
            phase_duration = (end - start) / self.sampling_rate

            # Classify phase pattern
            power_norm = (phase_power - np.min(phase_power)) / (np.max(phase_power) - np.min(phase_power) + 1e-7)
            pattern_type = self._classify_pattern(power_norm)

            phases.append(
                {
                    "phase_id": i,
                    "start_sec": float(start / self.sampling_rate),
                    "end_sec": float(end / self.sampling_rate),
                    "duration_sec": float(phase_duration),
                    "pattern_type": pattern_type,
                    "avg_power": float(np.mean(phase_power)),
                    "std_power": float(np.std(phase_power)),
                }
            )

        return phases
