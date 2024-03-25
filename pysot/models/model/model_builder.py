

from pysot.models.model.siamrpnpp_builder import SiamRPNppBuilder
from pysot.models.model.siamatt_builder import SiamAttBuilder
from pysot.models.model.siamban_builder import SiamBANBuilder
from pysot.models.model.siamcar_builder import SiamCARBuilder
from pysot.models.model.siamcarm_builder import SiamCARMBuilder
from pysot.models.model.siamgat_builder import SiamGATBuilder
from pysot.models.model.transt_builder import TransTBuilder
from pysot.models.model.mobilesiam_builder import MobileSiamBuilder
from pysot.models.model.updmobilesiam_builder import UPDMobileSiamBuilder


MODELS = {
        'MobileSiam': MobileSiamBuilder,
        'UPDMobileSiam': UPDMobileSiamBuilder,
        'SiamRPNpp': SiamRPNppBuilder,
        'SiamMask': SiamRPNppBuilder,
        'SiamAtt': SiamAttBuilder,
        'SiamBAN': SiamBANBuilder,
        'SiamCAR': SiamCARBuilder,
        'SiamCARM': SiamCARMBuilder,
        'SiamGAT': SiamGATBuilder,
        'TransT': TransTBuilder
        }


def build_model(name, cfg):
    return MODELS[name](cfg)
