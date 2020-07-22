from base.Base import Base
from base.ABCModel import ABCModel
from base.BaseLogger import BaseLogger
from attrdict import AttrDict
from abc import *


class ABCModelRunner(Base, metaclass=ABCMeta):
    def __init__(self, conf, logger: BaseLogger, model: ABCModel):
        super(ABCModelRunner, self).__init__(conf, logger)
        self.model = model

    @abstractmethod
    def run(self, x=None, y=None, val_x=None, val_y=None, *args, **kwargs):
        raise NotImplementedError

    def getModel(self):
        return self.model.model
