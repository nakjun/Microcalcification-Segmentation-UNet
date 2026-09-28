"""Study-cluster intervals of fold-averaged paired differences."""
import numpy as np

FIELDS = ('positive_macro_Dice', 'pixel_recall', 'negative_FP_pixels', 'negative_images_with_FP')

def vector(c):
    positive = c['gt_pixels'] > 0
    assert c['TP'] + c['FN'] == c['gt_pixels']
    assert c['TP'] + c['FP'] == c['predicted_pixels']
    return [int(positive), c['Dice'] if positive else 0., c['TP'], c['FN'],
            c['FP'] if not positive else 0, int(not positive and c['FP'] > 0)]

def metrics(x):
    def div(a, b):
        return np.divide(a, b, out=np.full_like(a, np.nan, dtype=float), where=b != 0)
    return np.stack([div(x[..., 1], x[..., 0]), div(x[..., 2], x[..., 2] + x[..., 3]),
                     x[..., 4], x[..., 5]], axis=-1)

def bootstrap(cases, studies, replicates=2000, seed=314159):
    # cases: [condition=H/LH, fold, image, additive statistic]
    x = np.asarray(cases, dtype=float)
    assert x.shape[:2] == (2, 3) and x.shape[2] == len(studies)
    unique = sorted(set(studies))
    grouped = np.stack([x[:, :, np.asarray(studies) == s, :].sum(axis=2) for s in unique], axis=2)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(unique), size=(replicates, len(unique)))
    weights = np.asarray([np.bincount(d, minlength=len(unique)) for d in draws])
    sums = np.einsum('rs,cfsk->rcfk', weights, grouped)
    values = metrics(sums)
    delta = (values[:, 1] - values[:, 0]).mean(axis=1)
    point = (metrics(x.sum(axis=2))[1] - metrics(x.sum(axis=2))[0]).mean(axis=0)
    result = {}
    for i, field in enumerate(FIELDS):
        valid = delta[np.isfinite(delta[:, i]), i]
        result[field] = dict(estimate=float(point[i]), lower=float(np.percentile(valid, 2.5)),
                             upper=float(np.percentile(valid, 97.5)), valid_replicates=len(valid))
    return result

def self_test():
    x = np.array([[[[1, .5, 2, 2, 0, 0], [0, 0, 0, 0, 4, 1]]] * 3] * 2, dtype=float)
    result = bootstrap(x, ['a', 'b'], replicates=100)
    assert all(r['estimate'] == r['lower'] == r['upper'] == 0 for r in result.values())
    assert result['positive_macro_Dice']['valid_replicates'] < 100

if __name__ == '__main__':
    self_test()
    print('paired fold-mean bootstrap test passed')
