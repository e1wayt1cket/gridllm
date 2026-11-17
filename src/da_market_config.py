"""Load the one operational research declaration for the independent DA market."""

import json

from da_market_data import MarketRules,MECHANISM_VERSION
from run_local_da import ROOT


def default_rules():
    data=json.loads((ROOT/'config/independent_da.json').read_text(encoding='utf-8'))
    if data['mechanism_version']!=MECHANISM_VERSION:
        raise ValueError('Configuration mechanism version mismatch')
    return MarketRules(**{k:data[k] for k in MarketRules.__dataclass_fields__})
