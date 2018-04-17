import os

from tqdm import tqdm

from data_feeder import CellImageDataManagerTest
from submission import KaggleSubmission

if __name__ == '__main__':
    tag = 'default'
    model = 'stage2_unetv1_norcnn'     # TODO : to be changed

    all_set = set(CellImageDataManagerTest.LIST)
    id_set = set()
    lines = ['ImageId,EncodedPixels']

    print(model)
    for seg in tqdm(list(range(19))):
        start_idx = 160 * seg
        end_idx = 160 * (seg + 1)
        name = 'ensemble_%s_%s_(%d_%d)' % (tag, model, start_idx, end_idx)

        filepath = os.path.join(KaggleSubmission.BASEPATH, name, 'submission_%s.csv' % name)

        f = open(filepath, 'r')
        skip_first = True
        for line in f.readlines():
            if skip_first:
                skip_first = False
                continue
            lines.append(line.strip())

            id_set.add(line.split(',')[0])

        f.close()

    f = open('./submissions_stage2/%s_merged.csv' % model, 'w')
    f.write('\n'.join(lines))
    f.close()

    print('size=', len(all_set), len(id_set))
    assert all_set == id_set, all_set - id_set
