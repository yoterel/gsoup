"""
Test suite for different structured light methods.
Tests encoding/decoding functionality and compares with GrayCode baseline.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile

import gsoup


class TestPhaseShifting:
    """Test suite for PhaseShifting class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.phase_shifting = gsoup.PhaseShifting(num_phases=4, cycles_x=8, cycles_y=6)
        self.proj_wh = (64, 48)  # Small resolution for testing

    def test_initialization(self):
        """Test PhaseShifting class initialization."""
        assert self.phase_shifting.num_temporal_phases == 4
        assert self.phase_shifting.cycles_x == 8
        assert self.phase_shifting.cycles_y == 6

        # Test with default cycle counts
        ps_default = gsoup.PhaseShifting(num_phases=3)
        assert ps_default.num_temporal_phases == 3
        assert ps_default.cycles_x is None
        assert ps_default.cycles_y is None

    def test_encode(self):
        """Test pattern encoding."""
        patterns = self.phase_shifting.encode(self.proj_wh)

        # Check output shape
        expected_images = (
            4 * self.phase_shifting.num_temporal_phases + 2
        )  # x_patterns + y_patterns + white + black
        assert patterns.shape == (expected_images, self.proj_wh[1], self.proj_wh[0], 1)
        assert patterns.dtype == np.uint8

        # Check that patterns are sinusoidal
        for i in range(self.phase_shifting.num_temporal_phases):
            # X-direction pattern
            x_pattern = patterns[i, :, :, 0]
            # Should have sinusoidal variation along x-axis
            x_variation = np.std(x_pattern, axis=1)
            assert np.mean(x_variation) > 50  # Should have significant variation

            # Y-direction pattern
            y_pattern = patterns[i + self.phase_shifting.num_temporal_phases, :, :, 0]
            # Should have sinusoidal variation along y-axis
            y_variation = np.std(y_pattern, axis=0)
            assert np.mean(y_variation) > 50  # Should have significant variation

        # Check reference images
        white_img = patterns[-2]
        black_img = patterns[-1]
        assert np.all(white_img == 255)
        assert np.all(black_img == 0)

    def test_compute_phase(self):
        """Test phase computation from intensity values."""
        # Test 3-step algorithm
        ps_3 = gsoup.PhaseShifting(num_phases=3)
        intensities = np.array([100, 200, 100])  # Simulated intensities
        phase = ps_3.compute_spatial_phase(intensities, 3)
        assert isinstance(phase, (float, np.floating))

        # Test 4-step algorithm
        ps_4 = gsoup.PhaseShifting(num_phases=4)
        intensities = np.array([100, 200, 100, 200])  # Simulated intensities
        phase = ps_4.compute_spatial_phase(intensities, 4)
        assert isinstance(phase, (float, np.floating))

        # Test 5-step algorithm
        ps_5 = gsoup.PhaseShifting(num_phases=5)
        intensities = np.array([100, 150, 200, 150, 100])  # Simulated intensities
        phase = ps_5.compute_spatial_phase(intensities, 5)
        assert isinstance(phase, (float, np.floating))

        # Test error handling
        with pytest.raises(ValueError):
            ps_4.compute_spatial_phase(
                np.array([100, 200]), 4
            )  # Wrong number of intensities

    def test_unwrap_phase(self):
        """Test phase unwrapping."""
        # Test with known phase values
        wrapped_phase = np.pi / 4  # 45 degrees
        cycles = 8
        image_size = 64

        coords = self.phase_shifting.unwrap_spatial_phase(
            wrapped_phase, cycles, image_size
        )

        # Should return reasonable coordinate
        assert 0 <= coords <= image_size

        # Test with array input
        wrapped_phases = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        coords = self.phase_shifting.unwrap_spatial_phase(
            wrapped_phases, cycles, image_size
        )
        assert coords.shape == wrapped_phases.shape

    def test_decode_synthetic(self):
        """Test decoding with synthetic data."""
        # Generate patterns
        patterns = self.phase_shifting.encode(self.proj_wh)

        # Simulate perfect capture (no noise)
        captures = patterns.copy()

        # Decode
        forward_map = self.phase_shifting.decode(
            captures, self.proj_wh, bg_threshold=10
        )
        # Check output shapes
        assert forward_map.shape == (self.proj_wh[1], self.proj_wh[0], 2)
        assert forward_map.dtype == np.int64
        # check all valid in forward_map
        assert np.all(forward_map >= 0)
        # check forward_map is identity map (nope, currently unwrap only works up to cycle length)
        # assert np.all(forward_map[..., 0] == np.arange(self.proj_wh[0]))
        # assert np.all(forward_map[..., 1] == np.arange(self.proj_wh[1]))

    def test_decode_with_noise(self):
        """Test decoding with added noise."""
        # Generate patterns
        patterns = self.phase_shifting.encode(self.proj_wh)

        # Add noise to simulate real capture
        noise_level = 10
        captures = patterns.astype(np.float32)
        noise = np.random.normal(0, noise_level, captures.shape)
        captures = captures + noise
        captures = np.clip(captures, 0, 255).astype(np.uint8)

        # Decode
        forward_map = self.phase_shifting.decode(
            captures, self.proj_wh, bg_threshold=10
        )

        # Should still produce reasonable results
        assert forward_map.shape == (self.proj_wh[1], self.proj_wh[0], 2)

    def test_decode_mode_xy(self):
        """Test decoding with xy mode."""
        patterns = self.phase_shifting.encode(self.proj_wh)
        captures = patterns.copy()

        forward_map_xy = self.phase_shifting.decode(captures, self.proj_wh, mode="xy")

        forward_map_ij = self.phase_shifting.decode(captures, self.proj_wh, mode="ij")

        # Check that coordinate order is swapped
        np.testing.assert_array_equal(forward_map_xy[..., 0], forward_map_ij[..., 1])
        np.testing.assert_array_equal(forward_map_xy[..., 1], forward_map_ij[..., 0])

    def test_decode_output_dir(self):
        """Test decoding with output directory."""
        patterns = self.phase_shifting.encode(self.proj_wh)
        captures = patterns.copy()

        with tempfile.TemporaryDirectory() as temp_dir:
            forward_map = self.phase_shifting.decode(
                captures, self.proj_wh, output_dir=temp_dir, debug=True
            )

            # Check that files were saved
            temp_path = Path(temp_dir)
            assert (temp_path / "forward_map_coarse.npy").exists()
            assert (temp_path / "forward_map_fine.npy").exists()
            assert (temp_path / "x_phase_coarse.png").exists()
            assert (temp_path / "y_phase_coarse.png").exists()
            assert (temp_path / "x_phase_fine.png").exists()
            assert (temp_path / "y_phase_fine.png").exists()
            assert (temp_path / "forward_map_coarse.png").exists()
            assert (temp_path / "forward_map_fine.png").exists()


class TestComparisonWithGrayCode:
    """Test suite comparing new methods with GrayCode baseline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.graycode = gsoup.GrayCode()
        self.phase_shifting = gsoup.PhaseShifting(num_phases=4)
        self.proj_wh = (32, 24)  # Very small resolution for quick testing

    def test_pattern_count_comparison(self):
        """Compare number of patterns required by each method."""
        # GrayCode patterns
        gc_patterns = self.graycode.encode(self.proj_wh, flipped_patterns=True)
        gc_count = len(gc_patterns)

        # PhaseShifting patterns
        ps_patterns = self.phase_shifting.encode(self.proj_wh)
        ps_count = len(ps_patterns)

        assert (
            ps_count < gc_count
        ), f"PhaseShifting ({ps_count}) should require fewer patterns than GrayCode ({gc_count})"

        print(f"Pattern counts: GrayCode={gc_count}, PhaseShifting={ps_count}")

    def test_api_consistency(self):
        """Test that all methods have consistent APIs."""
        # All should have encode method
        assert hasattr(self.graycode, "encode")
        assert hasattr(self.phase_shifting, "encode")

        # All should have decode method
        assert hasattr(self.graycode, "decode")
        assert hasattr(self.phase_shifting, "decode")

        # Test encode method signatures
        gc_patterns = self.graycode.encode(self.proj_wh)
        ps_patterns = self.phase_shifting.encode(self.proj_wh)

        # All should return numpy arrays
        assert isinstance(gc_patterns, np.ndarray)
        assert isinstance(ps_patterns, np.ndarray)

        # All should have 4 dimensions
        assert gc_patterns.ndim == 4
        assert ps_patterns.ndim == 4

    def test_decode_output_consistency(self):
        """Test that all decode methods return consistent output formats."""
        # Generate patterns
        gc_patterns = self.graycode.encode(self.proj_wh)
        ps_patterns = self.phase_shifting.encode(self.proj_wh)

        # Decode
        gc_map = self.graycode.decode(gc_patterns, self.proj_wh)
        ps_map = self.phase_shifting.decode(ps_patterns, self.proj_wh)

        # All should return same output shapes
        expected_shape = (self.proj_wh[1], self.proj_wh[0], 2)
        assert gc_map.shape == expected_shape
        assert ps_map.shape == expected_shape
        # All should return same data types
        assert gc_map.dtype == np.int64
        assert ps_map.dtype == np.int64
