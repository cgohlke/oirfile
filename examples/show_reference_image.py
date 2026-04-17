"""Example: show main image, reference image, and scanned line ROI."""

from __future__ import annotations

import argparse

from matplotlib import pyplot

from oirfile import OirFile


def select_display_plane(data, dims):
    """Return a 2D plane suitable for display with pyplot.imshow."""
    if data.ndim == 2:
        return data, '2D image'

    if data.ndim == 3:
        if dims == ('T', 'Y', 'X'):
            return data[0], 'first time point'
        if dims == ('C', 'Y', 'X'):
            return data[0], 'first channel'
        return data[0], f'first plane along {dims[0]}'

    raise ValueError(
        f'Cannot display primary image with shape {data.shape} and dims {dims}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Show OIR primary image, reference image, and line ROI.'
    )
    parser.add_argument('path', help='Path to OIR file')
    args = parser.parse_args()

    with OirFile(args.path) as oir:
        img_data = oir.asarray()
        print(oir)
        print('shape:', oir.shape)
        print('dims:', oir.dims)
        print('axis_kinds:', oir.axis_kinds)
        print('axis_units:', oir.axis_units)
        print('has_reference:', oir.has_reference)
        print('reference_shape:', oir.reference_shape)
        print('line_coordinates:', oir.line_coordinates)
        print('reference_axis_units:', oir.reference_axis_units)
        print('reference_pixel_size:', oir.reference_pixel_size)
        print('line_coordinates_physical:', oir.line_coordinates_physical)

        display_data, display_label = select_display_plane(img_data, oir.dims)

        pyplot.figure()
        pyplot.imshow(display_data, aspect='auto')
        pyplot.title(f'Primary image ({display_label})')
        pyplot.colorbar()

        if len(oir.dims) == 2:
            pyplot.xlabel(f"X ({oir.axis_units.get('X', '')})")
            pyplot.ylabel(f"Y ({oir.axis_units.get('Y', '')})")
        elif len(oir.dims) == 3 and oir.dims[-2:] == ('Y', 'X'):
            pyplot.xlabel(f"X ({oir.axis_units.get('X', '')})")
            pyplot.ylabel(f"Y ({oir.axis_units.get('Y', '')})")

        if oir.has_reference:
            ref = oir.asarray_reference()

            pyplot.figure()
            pyplot.imshow(ref)
            if oir.line_coordinates is not None:
                (x0, y0), (x1, y1) = oir.line_coordinates
                pyplot.plot([x0, x1], [y0, y1], linewidth=2)
                pyplot.scatter([x0, x1], [y0, y1], s=20)
            pyplot.title('Reference image with scanned line ROI')
            pyplot.colorbar()

        pyplot.show()


if __name__ == '__main__':
    main()