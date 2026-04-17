"""Example: show main line-scan image, reference image, and scanned line ROI."""

from __future__ import annotations

import argparse

from matplotlib import pyplot
from PIL import Image
import io

from oirfile import OirFile

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to OIR file')
    args = parser.parse_args()

    with OirFile(args.path) as oir:
        data = oir.asarray()
        print('path:', args.path)
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
        print('has_bmp:', oir.has_bmp)
        print('bmp_count:', oir.bmp_count)
        print('bmp_shape:', oir.bmp_shape)

        # if oir.has_bmp:
        #     bmp_bytes = oir.asbytes_bmp()
        #     print('bmp_bytes:', len(bmp_bytes))

        if not oir.has_reference:
            raise SystemExit('No reference image found')

        ref = oir.asarray_reference()

        pyplot.figure()
        pyplot.imshow(ref)
        if oir.line_coordinates is not None:
            (x0, y0), (x1, y1) = oir.line_coordinates
            pyplot.plot([x0, x1], [y0, y1], linewidth=2)
            pyplot.scatter([x0, x1], [y0, y1], s=20)
        pyplot.title('Reference image with scanned line ROI')
        pyplot.colorbar()

        pyplot.figure()
        pyplot.imshow(data, aspect='auto')
        pyplot.title('Primary line-scan image')
        pyplot.xlabel(f"X ({oir.axis_units.get('X', '')})")
        pyplot.ylabel(f"Y ({oir.axis_units.get('Y', '')})")
        pyplot.colorbar()

        # bmp_img = Image.open(io.BytesIO(bmp_bytes))
        # pyplot.figure()
        # pyplot.imshow(bmp_img)
        # pyplot.title('BMP image')
        # pyplot.colorbar()


        pyplot.show()


if __name__ == '__main__':
    main()
