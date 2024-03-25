

from configs.siamrpnpp_config import siamrpnpp_cfg
from configs.siamatt_config import siamatt_cfg
from configs.siamban_config import siamban_cfg
from configs.siamcar_config import siamcar_cfg
from configs.siamcarm_config import siamcarm_cfg
from configs.siamgat_config import siamgat_cfg
from configs.transt_config import transt_cfg
from configs.mobilesiam_config import mobilesiam_cfg
from configs.updmobilesiam_config import updmobilesiam_cfg

class Config(object):
    def __init__(self):
        pass


CONFIGS = {
          'MobileSiam': mobilesiam_cfg,
          'UPDMobileSiam': updmobilesiam_cfg,
          'SiamRPNpp': siamrpnpp_cfg,
          'SiamMask': siamrpnpp_cfg,
          'SiamAtt': siamatt_cfg,
          'SiamBAN': siamban_cfg,
          'SiamCAR': siamcar_cfg,
          'SiamCARM': siamcarm_cfg,
          'SiamGAT': siamgat_cfg,
          'TransT': transt_cfg
          }


def get_config(name):
    return CONFIGS[name]
