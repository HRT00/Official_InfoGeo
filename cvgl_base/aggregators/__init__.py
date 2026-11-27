from .cosplace import CosPlace
from .convap import ConvAP
from .gem import GeMPool
from .mixvpr import MixVPR
from .netvlad import NetVLAD
def __getattr__(name):
    if name == "SuperVLAD":
        from .supervlad import SuperVLAD
        return SuperVLAD
    raise AttributeError(name)
