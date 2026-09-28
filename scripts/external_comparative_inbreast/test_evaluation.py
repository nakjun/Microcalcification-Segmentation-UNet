"""Independent synthetic checks for the annotation-limited endpoint definitions."""
import numpy as np

from evaluate import image_metrics, summarize, fold_range


def test_point_contour_metrics():
    mask = np.zeros((7, 7), bool); mask[0, 0] = 1; mask[3, 3] = 1
    annotation = dict(points=[dict(roi_id='a', x=0, y=1), dict(roi_id='b', x=3, y=3), dict(roi_id='c', x=6, y=6)],
                      contours=[dict(roi_id='d', raster_xy=[[2, 2], [4, 2], [4, 4], [2, 4]])])
    result = image_metrics(mask, annotation)
    assert (result['points_exact'], result['points_within1']) == (1, 2)
    assert result['contour_union_pixels'] == 9 and result['contour_overlap_pixels'] == 1
    assert result['partial_contour_Dice'] == 2 / 11 and result['partial_contour_IoU'] == .1
    assert result['predicted_pixels_outside_contour_union_not_confirmed_FP'] == 1


def test_summary_and_fold_range():
    empty = dict(predicted_pixels=0, predicted_fraction=0., predicted_components=0, image_predicted_positive=False,
                 points_total=0, points_exact=0, points_within1=0, contours_total=0, contours_any_overlap=0,
                 contour_union_pixels=0, contour_overlap_pixels=0, partial_contour_Dice=None, partial_contour_IoU=None,
                 predicted_pixels_outside_contour_union_not_confirmed_FP=None)
    assert summarize([dict(policies=dict(primary=empty))], 'primary')['partial_contour_macro_Dice'] is None
    folds = [dict(metrics=dict(primary=dict(point_exact_hit_rate=value, predicted_pixels=number)))
             for value, number in ((.1, 2), (.2, 4), (.3, 3))]
    values = fold_range(folds, 'primary')
    np.testing.assert_allclose([values['point_exact_hit_rate'][k] for k in ('mean', 'min', 'max')], [.2, .1, .3])
    np.testing.assert_allclose([values['predicted_pixels'][k] for k in ('mean', 'min', 'max')], [3., 2., 4.])


if __name__ == '__main__':
    test_point_contour_metrics(); test_summary_and_fold_range()
    print('evaluation synthetic tests passed')
