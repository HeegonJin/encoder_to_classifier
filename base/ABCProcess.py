from base.BaseLogger import BaseLogger
from base.Base import Base
from attrdict import AttrDict
from abc import *


class ABCProcess(Base, metaclass=ABCMeta):
    def __init__(self, conf: AttrDict, logger: BaseLogger):
        super(ABCProcess, self).__init__(conf, logger)

    @abstractmethod
    def run(self, data=None):
        raise NotImplementedError
