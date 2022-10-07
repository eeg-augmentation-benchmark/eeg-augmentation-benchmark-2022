
from BAE.class_wise_aug import prepare_aug_list
from BAE.utils import ClassWiseAugmentation


def test_get_CW_augs():
    aug_list = prepare_aug_list(n_gpus=1)
    n_augs = len(aug_list[0])
    # 2**5 - 2 --> for every pair of augmentation we remove the 2
    # augmentatiosn that are class agnostic.
    assert n_augs == 1650  # (2**5 - 2) * math.comb(11, 2)
    for i in range(n_augs):
        assert isinstance(aug_list[0][i], ClassWiseAugmentation)
