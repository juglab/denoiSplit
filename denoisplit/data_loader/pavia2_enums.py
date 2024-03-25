from denoisplit.core.custom_enum import Enum

class Pavia2DataSetType(Enum):
    JustCYAN = '0b001'
    JustMAGENTA = '0b010'
    MIXED = '0b100'


class Pavia2DataSetChannels(Enum):
    NucRFP670 = 0
    NucMTORQ = 1
    ACTIN = 2
    TUBULIN = 3


class Pavia2DataSetVersion(Enum):
    DD = 'DenoisedDeconvolved'
    RAW = 'Raw data'

class Pavia2BleedthroughType(Enum):
    Clean = 0
    Bleedthrough = 1
    Mixed = 2