from base.Base import Base
from abc import *


class ABCModel(Base, metaclass=ABCMeta):
    def __init__(self, conf, logger):
        super(ABCModel, self).__init__(conf, logger)
        self.model = self.buildModel()

    @abstractmethod
    def buildModel(self):
        raise NotImplementedError

