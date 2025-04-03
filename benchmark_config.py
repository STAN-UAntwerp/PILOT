UCI_DATASET_IDS = [
    1,
    9,
    10,
    # 16, # classification target
    # 29, # no target defined
    60,
    87,
    # 89, # 3 targets
    # 92, # constant target
    162,
    165,
    183,
    186,
    # 189, # 2 targets
    # 211, # 18 targets
    # 235, # no target defined
    # 242, # 2 targets
    # 244, # binary target
    # 247, # no target defined
    # 270, # classification target
    275,
    291,
    294,
    # 300, # binary target
    # 320, # 3 targets
    332,  # SKipping because takes too long
    # 360, # no target defined
    368,
    374,
    381,
    # 390, # 6 targets
    409,
    # 461, # no target defined
    # 462, # loading error
    # 464, # loading error
    # 471, # 2 targets
    477,
    492,
    # 519, # binary target
    # 536, # no target defined
    # 544, # classification target
    # 547,  # binary target
    # 551, # no target defined
    # 555, # no target defined
    # 560, # binary target
    # 563, # binary target
    # 565, # binary target
    597,
    # 601, # 6 targets
    # 713, # 2 targets
    # 760, # loading error
    # 849, # 3 targets
    # 851, # loading error
    # 857, # binary target
    # 880, # 3 targets
    # 890, # binary target
    # 913, # no target defined
    # 925, # 2 targets
    # 942, # classification target
]

IGNORE_PMLB = ['banana', 'titanic']
IGNORE_COLUMNS = {183: ["state", "county", "community", "communityname", "fold"]}
LOGTRANSFORM_TARGET = [162, 332]
