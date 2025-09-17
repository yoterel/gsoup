"""
Test suite for DeBruijn and PhaseShifting pattern decoding classes.
Tests encoding/decoding functionality and compares with GrayCode baseline.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from gsoup.procam import GrayCode, DeBruijn, PhaseShifting
from gsoup.gsoup_io import save_image, load_image


class TestDeBruijn:
    """Test suite for DeBruijn class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.debruijn = DeBruijn(alphabet_size=4, sequence_length=3)
        self.proj_wh = (64, 48)  # Small resolution for testing
        
    def test_initialization(self):
        """Test DeBruijn class initialization."""
        assert self.debruijn.alphabet_size == 4
        assert self.debruijn.sequence_length == 3
        assert self.debruijn.colors.shape == (4, 3)
        
        # Test color palette
        expected_colors = np.array([
            [255, 0, 0],    # Red
            [0, 255, 0],    # Green
            [0, 0, 255],    # Blue
            [255, 255, 255] # White
        ], dtype=np.uint8)
        np.testing.assert_array_equal(self.debruijn.colors, expected_colors)
        
    def test_generate_debruijn_sequence(self):
        """Test De Bruijn sequence generation."""
        # Test simple case
        seq = self.debruijn.generate_debruijn_sequence(2, 2)
        assert len(seq) == 4  # 2^2 = 4
        assert seq.dtype == np.uint8
        
        # Test that sequence contains all possible 2-symbol combinations
        # For n=2, k=2: should contain 00, 01, 10, 11
        combinations = set()
        for i in range(len(seq) - 1):
            combo = tuple(seq[i:i+2])
            combinations.add(combo)
        assert len(combinations) == 4
        
        # Test larger case
        seq = self.debruijn.generate_debruijn_sequence(3, 4)
        assert len(seq) == 64  # 4^3 = 64
        assert seq.dtype == np.uint8
        
    def test_encode(self):
        """Test pattern encoding."""
        patterns = self.debruijn.encode(self.proj_wh, include_reference=True)
        
        # Check output shape
        expected_images = 5  # x_pattern, y_pattern, combined_pattern, white, black
        assert patterns.shape == (expected_images, self.proj_wh[1], self.proj_wh[0], 3)
        assert patterns.dtype == np.uint8
        
        # Check that patterns contain valid colors
        for i in range(expected_images):
            pattern = patterns[i]
            # All pixel values should be in the color palette
            for y in range(pattern.shape[0]):
                for x in range(pattern.shape[1]):
                    pixel = pattern[y, x]
                    assert np.any(np.all(self.debruijn.colors == pixel, axis=1))
                    
        # Check reference images
        white_img = patterns[-2]
        black_img = patterns[-1]
        assert np.all(white_img == 255)
        assert np.all(black_img == 0)
        
    def test_decode_color_sequence(self):
        """Test color sequence decoding."""
        # Test with exact color match
        red_patch = np.array([255, 0, 0])
        idx, valid = self.debruijn.decode_color_sequence(red_patch)
        assert idx == 0  # Red is index 0
        assert valid == True
        
        # Test with approximate color match
        red_patch_noisy = np.array([250, 5, 5])
        idx, valid = self.debruijn.decode_color_sequence(red_patch_noisy, color_tolerance=20)
        assert idx == 0
        assert valid == True
        
        # Test with invalid color
        invalid_patch = np.array([100, 100, 100])
        idx, valid = self.debruijn.decode_color_sequence(invalid_patch, color_tolerance=20)
        assert valid == False
        
    def test_decode_synthetic(self):
        """Test decoding with synthetic data."""
        # Generate patterns
        patterns = self.debruijn.encode(self.proj_wh, include_reference=True)
        
        # Simulate perfect capture (no noise)
        captures = patterns.copy()
        
        # Decode
        forward_map, fg = self.debruijn.decode(
            captures, self.proj_wh, color_tolerance=50, bg_threshold=10
        )
        
        # Check output shapes
        assert forward_map.shape == (self.proj_wh[1], self.proj_wh[0], 2)
        assert fg.shape == (self.proj_wh[1], self.proj_wh[0])
        assert forward_map.dtype == np.uint32
        assert fg.dtype == bool
        
        # Check that foreground mask is reasonable
        assert np.any(fg)  # Should have some foreground pixels
        
    def test_decode_with_noise(self):
        """Test decoding with added noise."""
        # Generate patterns
        patterns = self.debruijn.encode(self.proj_wh, include_reference=True)
        
        # Add noise to simulate real capture
        noise_level = 20
        captures = patterns.astype(np.float32)
        noise = np.random.normal(0, noise_level, captures.shape)
        captures = captures + noise
        captures = np.clip(captures, 0, 255).astype(np.uint8)
        
        # Decode with higher tolerance
        forward_map, fg = self.debruijn.decode(
            captures, self.proj_wh, color_tolerance=100, bg_threshold=10
        )
        
        # Should still produce reasonable results
        assert forward_map.shape == (self.proj_wh[1], self.proj_wh[0], 2)
        assert fg.shape == (self.proj_wh[1], self.proj_wh[0])
        
    def test_decode_mode_xy(self):
        """Test decoding with xy mode."""
        patterns = self.debruijn.encode(self.proj_wh, include_reference=True)
        captures = patterns.copy()
        
        forward_map_xy, fg_xy = self.debruijn.decode(
            captures, self.proj_wh, mode="xy"
        )
        
        forward_map_ij, fg_ij = self.debruijn.decode(
            captures, self.proj_wh, mode="ij"
        )
        
        # Check that coordinate order is swapped
        np.testing.assert_array_equal(
            forward_map_xy[..., 0], forward_map_ij[..., 1]
        )
        np.testing.assert_array_equal(
            forward_map_xy[..., 1], forward_map_ij[..., 0]
        )
        
    def test_decode_output_dir(self):
        """Test decoding with output directory."""
        patterns = self.debruijn.encode(self.proj_wh, include_reference=True)
        captures = patterns.copy()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            forward_map, fg = self.debruijn.decode(
                captures, self.proj_wh, output_dir=temp_dir, debug=True
            )
            
            # Check that files were saved
            temp_path = Path(temp_dir)
            assert (temp_path / "forward_map.npy").exists()
            assert (temp_path / "fg.npy").exists()
            assert (temp_path / "x_coords.png").exists()
            assert (temp_path / "y_coords.png").exists()
            assert (temp_path / "foreground.png").exists()
            assert (temp_path / "forward_map.png").exists()


class TestPhaseShifting:
    """Test suite for PhaseShifting class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.phase_shifting = PhaseShifting(num_phases=4, frequency_x=8, frequency_y=6)
        self.proj_wh = (64, 48)  # Small resolution for testing
        
    def test_initialization(self):
        """Test PhaseShifting class initialization."""
        assert self.phase_shifting.num_phases == 4
        assert self.phase_shifting.frequency_x == 8
        assert self.phase_shifting.frequency_y == 6
        
        # Test with default frequencies
        ps_default = PhaseShifting(num_phases=3)
        assert ps_default.num_phases == 3
        assert ps_default.frequency_x is None
        assert ps_default.frequency_y is None
        
    def test_encode(self):
        """Test pattern encoding."""
        patterns = self.phase_shifting.encode(self.proj_wh, include_reference=True)
        
        # Check output shape
        expected_images = 2 * self.phase_shifting.num_phases + 2  # x_patterns + y_patterns + white + black
        assert patterns.shape == (expected_images, self.proj_wh[1], self.proj_wh[0], 1)
        assert patterns.dtype == np.uint8
        
        # Check that patterns are sinusoidal
        for i in range(self.phase_shifting.num_phases):
            # X-direction pattern
            x_pattern = patterns[i, :, :, 0]
            # Should have sinusoidal variation along x-axis
            x_variation = np.std(x_pattern, axis=0)
            assert np.mean(x_variation) > 50  # Should have significant variation
            
            # Y-direction pattern
            y_pattern = patterns[i + self.phase_shifting.num_phases, :, :, 0]
            # Should have sinusoidal variation along y-axis
            y_variation = np.std(y_pattern, axis=1)
            assert np.mean(y_variation) > 50  # Should have significant variation
            
        # Check reference images
        white_img = patterns[-2]
        black_img = patterns[-1]
        assert np.all(white_img == 255)
        assert np.all(black_img == 0)
        
    def test_compute_phase(self):
        """Test phase computation from intensity values."""
        # Test 3-step algorithm
        ps_3 = PhaseShifting(num_phases=3)
        intensities = np.array([100, 200, 100])  # Simulated intensities
        phase = ps_3.compute_phase(intensities, 3)
        assert isinstance(phase, (float, np.floating))
        
        # Test 4-step algorithm
        ps_4 = PhaseShifting(num_phases=4)
        intensities = np.array([100, 200, 100, 200])  # Simulated intensities
        phase = ps_4.compute_phase(intensities, 4)
        assert isinstance(phase, (float, np.floating))
        
        # Test 5-step algorithm
        ps_5 = PhaseShifting(num_phases=5)
        intensities = np.array([100, 150, 200, 150, 100])  # Simulated intensities
        phase = ps_5.compute_phase(intensities, 5)
        assert isinstance(phase, (float, np.floating))
        
        # Test error handling
        with pytest.raises(ValueError):
            ps_4.compute_phase(np.array([100, 200]), 4)  # Wrong number of intensities
            
    def test_unwrap_phase(self):
        """Test phase unwrapping."""
        # Test with known phase values
        wrapped_phase = np.pi / 4  # 45 degrees
        frequency = 8
        image_size = 64
        
        coords = self.phase_shifting.unwrap_phase(wrapped_phase, frequency, image_size)
        
        # Should return reasonable coordinate
        assert 0 <= coords <= image_size
        
        # Test with array input
        wrapped_phases = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
        coords = self.phase_shifting.unwrap_phase(wrapped_phases, frequency, image_size)
        assert coords.shape == wrapped_phases.shape
        
    def test_decode_synthetic(self):
        """Test decoding with synthetic data."""
        # Generate patterns
        patterns = self.phase_shifting.encode(self.proj_wh, include_reference=True)
        
        # Simulate perfect capture (no noise)
        captures = patterns.copy()
        
        # Decode
        forward_map, fg = self.phase_shifting.decode(
            captures, self.proj_wh, bg_threshold=10
        )
        
        # Check output shapes
        assert forward_map.shape == (self.proj_wh[1], self.proj_wh[0], 2)
        assert fg.shape == (self.proj_wh[1], self.proj_wh[0])
        assert forward_map.dtype == np.uint32
        assert fg.dtype == bool
        
        # Check that foreground mask is reasonable
        assert np.any(fg)  # Should have some foreground pixels
        
    def test_decode_with_noise(self):
        """Test decoding with added noise."""
        # Generate patterns
        patterns = self.phase_shifting.encode(self.proj_wh, include_reference=True)
        
        # Add noise to simulate real capture
        noise_level = 10
        captures = patterns.astype(np.float32)
        noise = np.random.normal(0, noise_level, captures.shape)
        captures = captures + noise
        captures = np.clip(captures, 0, 255).astype(np.uint8)
        
        # Decode
        forward_map, fg = self.phase_shifting.decode(
            captures, self.proj_wh, bg_threshold=10
        )
        
        # Should still produce reasonable results
        assert forward_map.shape == (self.proj_wh[1], self.proj_wh[0], 2)
        assert fg.shape == (self.proj_wh[1], self.proj_wh[0])
        
    def test_decode_mode_xy(self):
        """Test decoding with xy mode."""
        patterns = self.phase_shifting.encode(self.proj_wh, include_reference=True)
        captures = patterns.copy()
        
        forward_map_xy, fg_xy = self.phase_shifting.decode(
            captures, self.proj_wh, mode="xy"
        )
        
        forward_map_ij, fg_ij = self.phase_shifting.decode(
            captures, self.proj_wh, mode="ij"
        )
        
        # Check that coordinate order is swapped
        np.testing.assert_array_equal(
            forward_map_xy[..., 0], forward_map_ij[..., 1]
        )
        np.testing.assert_array_equal(
            forward_map_xy[..., 1], forward_map_ij[..., 0]
        )
        
    def test_decode_output_dir(self):
        """Test decoding with output directory."""
        patterns = self.phase_shifting.encode(self.proj_wh, include_reference=True)
        captures = patterns.copy()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            forward_map, fg = self.phase_shifting.decode(
                captures, self.proj_wh, output_dir=temp_dir, debug=True
            )
            
            # Check that files were saved
            temp_path = Path(temp_dir)
            assert (temp_path / "forward_map.npy").exists()
            assert (temp_path / "fg.npy").exists()
            assert (temp_path / "x_phase.png").exists()
            assert (temp_path / "y_phase.png").exists()
            assert (temp_path / "foreground.png").exists()
            assert (temp_path / "forward_map.png").exists()


class TestComparisonWithGrayCode:
    """Test suite comparing new methods with GrayCode baseline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.graycode = GrayCode()
        self.debruijn = DeBruijn(alphabet_size=4, sequence_length=3)
        self.phase_shifting = PhaseShifting(num_phases=4)
        self.proj_wh = (32, 24)  # Very small resolution for quick testing
        
    def test_pattern_count_comparison(self):
        """Compare number of patterns required by each method."""
        # GrayCode patterns
        gc_patterns = self.graycode.encode(self.proj_wh, flipped_patterns=True)
        gc_count = len(gc_patterns)
        
        # DeBruijn patterns
        db_patterns = self.debruijn.encode(self.proj_wh, include_reference=True)
        db_count = len(db_patterns)
        
        # PhaseShifting patterns
        ps_patterns = self.phase_shifting.encode(self.proj_wh, include_reference=True)
        ps_count = len(ps_patterns)
        
        # DeBruijn and PhaseShifting should require fewer patterns
        assert db_count < gc_count, f"DeBruijn ({db_count}) should require fewer patterns than GrayCode ({gc_count})"
        assert ps_count < gc_count, f"PhaseShifting ({ps_count}) should require fewer patterns than GrayCode ({gc_count})"
        
        print(f"Pattern counts: GrayCode={gc_count}, DeBruijn={db_count}, PhaseShifting={ps_count}")
        
    def test_api_consistency(self):
        """Test that all methods have consistent APIs."""
        # All should have encode method
        assert hasattr(self.graycode, 'encode')
        assert hasattr(self.debruijn, 'encode')
        assert hasattr(self.phase_shifting, 'encode')
        
        # All should have decode method
        assert hasattr(self.graycode, 'decode')
        assert hasattr(self.debruijn, 'decode')
        assert hasattr(self.phase_shifting, 'decode')
        
        # Test encode method signatures
        gc_patterns = self.graycode.encode(self.proj_wh)
        db_patterns = self.debruijn.encode(self.proj_wh)
        ps_patterns = self.phase_shifting.encode(self.proj_wh)
        
        # All should return numpy arrays
        assert isinstance(gc_patterns, np.ndarray)
        assert isinstance(db_patterns, np.ndarray)
        assert isinstance(ps_patterns, np.ndarray)
        
        # All should have 4 dimensions
        assert gc_patterns.ndim == 4
        assert db_patterns.ndim == 4
        assert ps_patterns.ndim == 4
        
    def test_decode_output_consistency(self):
        """Test that all decode methods return consistent output formats."""
        # Generate patterns
        gc_patterns = self.graycode.encode(self.proj_wh, flipped_patterns=True)
        db_patterns = self.debruijn.encode(self.proj_wh, include_reference=True)
        ps_patterns = self.phase_shifting.encode(self.proj_wh, include_reference=True)
        
        # Decode
        gc_map, gc_fg = self.graycode.decode(gc_patterns, self.proj_wh)
        db_map, db_fg = self.debruijn.decode(db_patterns, self.proj_wh)
        ps_map, ps_fg = self.phase_shifting.decode(ps_patterns, self.proj_wh)
        
        # All should return same output shapes
        expected_shape = (self.proj_wh[1], self.proj_wh[0], 2)
        assert gc_map.shape == expected_shape
        assert db_map.shape == expected_shape
        assert ps_map.shape == expected_shape
        
        # All should return same foreground mask shape
        expected_fg_shape = (self.proj_wh[1], self.proj_wh[0])
        assert gc_fg.shape == expected_fg_shape
        assert db_fg.shape == expected_fg_shape
        assert ps_fg.shape == expected_fg_shape
        
        # All should return same data types
        assert gc_map.dtype == np.uint32
        assert db_map.dtype == np.uint32
        assert ps_map.dtype == np.uint32
        
        assert gc_fg.dtype == bool
        assert db_fg.dtype == bool
        assert ps_fg.dtype == bool


def test_integration_example():
    """Integration test showing how to use the new classes."""
    # Example usage of DeBruijn
    print("\n=== DeBruijn Example ===")
    debruijn = DeBruijn(alphabet_size=4, sequence_length=3)
    proj_wh = (128, 96)
    
    # Generate patterns
    patterns = debruijn.encode(proj_wh, include_reference=True)
    print(f"Generated {len(patterns)} DeBruijn patterns")
    
    # Simulate capture (in real usage, these would come from camera)
    captures = patterns.copy()
    
    # Decode
    forward_map, fg = debruijn.decode(captures, proj_wh, mode="xy")
    print(f"Decoded forward map shape: {forward_map.shape}")
    print(f"Foreground pixels: {np.sum(fg)}")
    
    # Example usage of PhaseShifting
    print("\n=== PhaseShifting Example ===")
    phase_shifting = PhaseShifting(num_phases=4, frequency_x=16, frequency_y=12)
    
    # Generate patterns
    patterns = phase_shifting.encode(proj_wh, include_reference=True)
    print(f"Generated {len(patterns)} PhaseShifting patterns")
    
    # Simulate capture
    captures = patterns.copy()
    
    # Decode
    forward_map, fg = phase_shifting.decode(captures, proj_wh, mode="xy")
    print(f"Decoded forward map shape: {forward_map.shape}")
    print(f"Foreground pixels: {np.sum(fg)}")


if __name__ == "__main__":
    # Run the integration example
    test_integration_example()
    
    # Run pytest if available
    try:
        import pytest
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic tests...")
        
        # Run basic tests manually
        test_debruijn = TestDeBruijn()
        test_debruijn.setup_method()
        test_debruijn.test_initialization()
        test_debruijn.test_encode()
        test_debruijn.test_decode_synthetic()
        print("DeBruijn tests passed!")
        
        test_phase = TestPhaseShifting()
        test_phase.setup_method()
        test_phase.test_initialization()
        test_phase.test_encode()
        test_phase.test_decode_synthetic()
        print("PhaseShifting tests passed!")
        
        test_comparison = TestComparisonWithGrayCode()
        test_comparison.setup_method()
        test_comparison.test_pattern_count_comparison()
        test_comparison.test_api_consistency()
        print("Comparison tests passed!")
        
        print("All tests completed successfully!")
