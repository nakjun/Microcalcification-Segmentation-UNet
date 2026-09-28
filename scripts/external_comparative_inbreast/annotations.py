"""INbreast parser copied from external_v3 coordinate and validity rules."""
from __future__ import annotations

import ast
import plistlib
from collections import Counter

import cv2
import numpy as np

from common import INBREAST, ROOT, checksum, immutable


def build(rows):
    """This must only be called by evaluation after all image-only cohorts are complete."""
    inputs = {r['image_id']: r for r in rows}
    records, totals = [], Counter()
    sources = {str(INBREAST / 'README.txt'): checksum(INBREAST / 'README.txt')}
    for ident, source in sorted(inputs.items()):
        h, w = source['height'], source['width']
        xml = INBREAST / 'AllXML' / f'{ident}.xml'
        record = dict(image_id=ident, height=h, width=w, status='missing_xml', points=[], contours=[], issues=[])
        if xml.exists():
            sources[str(xml)] = checksum(xml)
            record.update(status='annotated_xml_partial_boundary', xml_path=str(xml), xml_sha256=sources[str(xml)])
            for image in plistlib.loads(xml.read_bytes()).get('Images', []):
                for index, roi in enumerate(image.get('ROIs', [])):
                    if roi.get('Name') != 'Calcification':
                        continue
                    points = np.asarray([ast.literal_eval(x) for x in roi.get('Point_px', [])], dtype=np.float64)
                    roi_id = str(roi.get('IndexInImage', index))
                    kind = 'point' if len(points) == 1 else 'contour' if len(points) >= 3 else 'degenerate'
                    totals['raw_' + kind] += 1
                    valid = (points.ndim == 2 and points.shape[1] == 2 and np.isfinite(points).all()
                             and len(points) == roi.get('NumberOfPoints')
                             and np.all((points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)))
                    raster = np.rint(points).astype(np.int64) if valid else np.zeros((0, 2), np.int64)
                    valid = valid and np.all((raster[:, 0] >= 0) & (raster[:, 0] < w) & (raster[:, 1] >= 0) & (raster[:, 1] < h))
                    if not valid or kind == 'degenerate':
                        record['issues'].append(dict(roi_id=roi_id, reason='invalid_coordinates_or_point_count'))
                        totals['excluded'] += 1
                    elif kind == 'point':
                        record['points'].append(dict(roi_id=roi_id, x=int(raster[0, 0]), y=int(raster[0, 1]), original_xy=points[0].tolist()))
                    elif abs(cv2.contourArea(points.astype(np.float32))) == 0:
                        record['issues'].append(dict(roi_id=roi_id, reason='zero_area_contour'))
                        totals['excluded'] += 1
                    else:
                        record['contours'].append(dict(roi_id=roi_id, points=points.tolist(), raster_xy=raster.tolist()))
            totals['xml_images'] += 1
        else:
            totals['missing_xml'] += 1
        totals['valid_points'] += len(record['points'])
        totals['valid_contours'] += len(record['contours'])
        records.append(record)
    value = dict(complete=True, dataset='inbreast', images=len(records), records=records, summary=dict(totals),
                 coordinate_rule='original DICOM orientation;np.rint ties-to-even;no clipping',
                 scope='points are not boundary GT;contour union is partial-annotation agreement;missing XML is unknown',
                 source_files=sources)
    immutable(ROOT / 'annotations/inbreast.json', value)
    return value
