from helpers.dataset_adapters.summe import SumMePseudoLabelAdapter
from helpers.dataset_adapters.tvsum import TVSumPseudoLabelAdapter


def get_pseudo_label_adapter(dataset_name: str):
    name = dataset_name.strip().lower()
    if name == 'tvsum':
        return TVSumPseudoLabelAdapter()
    if name == 'summe':
        return SumMePseudoLabelAdapter()
    raise ValueError(f'Invalid dataset name: {dataset_name}')