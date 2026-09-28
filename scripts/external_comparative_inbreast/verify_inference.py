"""Create a single completion marker only after all 48 full prediction cohorts verify."""
from common import ROOT, checksum, context, immutable, prediction_dir, verify


def main():
    c = context(); expected_ids = [row['image_id'] for row in c['rows']]; outputs = {}
    for checkpoint in c['checkpoints']:
        marker = prediction_dir(checkpoint) / 'predictions/complete.json'
        done = verify(marker, c['signature'])
        assert done['image_ids'] == expected_ids and done['smoke'] is False and done['GT_read'] is False
        assert done['checkpoint_sha256'] == checkpoint['checkpoint_sha256']
        outputs[str(marker)] = checksum(marker)
    immutable(ROOT / 'predictions/complete.json', dict(complete=True, signature=c['signature'], dataset='inbreast',
                                                        checkpoints=48, images_per_checkpoint=410, GT_read=False,
                                                        refinement_executed_all_images=True, output_files=outputs))
    print('verified 48 checkpoint cohorts x 410 images')


if __name__ == '__main__':
    main()
