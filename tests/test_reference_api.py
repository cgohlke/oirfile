"""Tests for reference-image and line-scan metadata APIs."""

from __future__ import annotations

import math
from pathlib import Path

import numpy
import pytest

from oirfile import OirFile


DATA = Path(__file__).resolve().parent / 'data'


@pytest.mark.parametrize(
    ('filename', 'shape', 'line_coordinates', 'reference_pixel_size'),
    [
        (
            '20251030_A106_0002.oir',
            (30000, 24),
            ((250, 236), (244, 214)),
            (0.27393082948859, 0.27393082948859),
        ),
        (
            '20251030_A106_0003.oir',
            (30000, 22),
            ((248, 248), (229, 239)),
            (0.158844873968623, 0.158844873968623),
        ),
        (
            '20251030_A106_0004.oir',
            (30000, 20),
            ((214, 194), (218, 175)),
            (0.158844873968623, 0.158844873968623),
        ),
    ],
)
def test_reference_api_linescan_files(
    filename: str,
    shape: tuple[int, int],
    line_coordinates: tuple[tuple[int, int], tuple[int, int]],
    reference_pixel_size: tuple[float, float],
) -> None:
    """Reference image and line ROI are exposed for line-scan OIR files."""
    with OirFile(DATA / filename) as oir:
        assert oir.shape == shape
        assert oir.dims == ('Y', 'X')

        assert oir.has_reference is True
        assert oir.reference_shape == (512, 512)
        assert oir.line_coordinates == line_coordinates

        ref = oir.asarray_reference()
        assert ref.shape == (512, 512)
        assert ref.dtype == numpy.dtype('uint16')
        assert int(ref.min()) >= 0
        assert int(ref.max()) > int(ref.min())

        assert oir.axis_kinds == {'Y': 'time', 'X': 'space'}
        assert oir.axis_units == {'Y': 'ms', 'X': 'um'}
        assert oir.reference_axis_units == {'Y': 'um', 'X': 'um'}
        assert oir.reference_pixel_size is not None
        assert oir.reference_pixel_size == pytest.approx(reference_pixel_size)

        assert oir.line_coordinates_physical is not None
        (x0, y0), (x1, y1) = line_coordinates
        (px0, py0), (px1, py1) = oir.line_coordinates_physical
        dy, dx = reference_pixel_size
        assert px0 == pytest.approx(x0 * dx)
        assert py0 == pytest.approx(y0 * dy)
        assert px1 == pytest.approx(x1 * dx)
        assert py1 == pytest.approx(y1 * dy)


@pytest.mark.parametrize(
    ('filename', 'expected_width'),
    [
        ('20251030_A106_0002.oir', 24),
        ('20251030_A106_0003.oir', 22),
        ('20251030_A106_0004.oir', 20),
    ],
)
def test_line_roi_length_matches_scan_width(
    filename: str,
    expected_width: int,
) -> None:
    """Line ROI length is consistent with the line-scan X dimension."""
    with OirFile(DATA / filename) as oir:
        assert oir.line_coordinates is not None
        (x0, y0), (x1, y1) = oir.line_coordinates
        length = math.hypot(x1 - x0, y1 - y0)
        assert abs(length - expected_width) < 2.0


def test_main_asarray_is_unchanged() -> None:
    """Primary image loading still returns the original line-scan array."""
    with OirFile(DATA / '20251030_A106_0002.oir') as oir:
        data = oir.asarray()
        assert data.shape == (30000, 24)
        assert data.dtype == numpy.dtype('uint16')
        assert int(data.min()) >= 0
        assert int(data.max()) > int(data.min())


def test_reference_api_timeseries_image_without_line_roi() -> None:
    """Reference image and axis metadata are exposed without a line ROI."""
    with OirFile(DATA / '20190416_b_0001.oir') as oir:
        assert oir.shape == (2, 512, 512)
        assert oir.dims == ('T', 'Y', 'X')

        assert oir.axis_kinds == {'T': 'time', 'Y': 'space', 'X': 'space'}
        assert oir.axis_units == {'T': 's', 'Y': 'um', 'X': 'um'}

        assert oir.has_reference is True
        assert oir.reference_shape == (512, 512)

        ref = oir.asarray_reference()
        assert ref.shape == (512, 512)
        assert ref.dtype == numpy.dtype('uint16')
        assert int(ref.min()) >= 0
        assert int(ref.max()) > int(ref.min())

        assert oir.line_coordinates is None
        assert oir.line_coordinates_physical is None

        assert oir.reference_axis_units == {'Y': 'um', 'X': 'um'}
        assert oir.reference_pixel_size == pytest.approx(
            (0.497184455521791, 0.497184455521791)
        )


def test_embedded_bmp_bytes() -> None:
    """Embedded BMP block can be returned as raw bytes."""
    with OirFile(DATA / '20251030_A106_0002.oir') as oir:
        assert oir.has_bmp is True
        assert oir.bmp_count == 1

        bmp = oir.asbytes_bmp()
        assert bmp[:2] == b'BM'
        assert len(bmp) > 100


def test_embedded_bmp_index_error() -> None:
    """Out-of-range BMP requests raise OirFileError."""
    from oirfile import OirFileError

    with OirFile(DATA / '20251030_A106_0002.oir') as oir:
        with pytest.raises(OirFileError):
            oir.asbytes_bmp(1)