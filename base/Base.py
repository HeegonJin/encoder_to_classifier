from abc import *


class Base(metaclass=ABCMeta):
    def __init__(self, conf, logger):
        self.conf = conf
        self.logger = logger

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def save(self):
        raise NotImplementedError
