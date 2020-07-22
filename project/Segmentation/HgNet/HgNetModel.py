import torch
from torch import nn
from base.ABCModel import ABCModel
import time
import os
from project.Segmentation.HgNet.Util.HgNet import HgNet
from project.Util import PytorchUtil


class HgNetModel(ABCModel):
    def buildModel(self):
        self.device = self.conf.device

        if 'cuda' not in self.device and 'cpu' not in self.device:
            raise ValueError('device setting error. model device is cuda or cpu')

        hgnet = HgNet(self.conf.output.classes)

        hgnet = hgnet.to(self.device)

        self.optimizer = torch.optim.__dict__[self.conf.optimizer.class_name](hgnet.parameters(),
                                                                              **self.conf.optimizer.config)
        self.loss = nn.CrossEntropyLoss().to(self.device)

        try:
            from apex import amp
            hardnet, self.optimizer = amp.initialize(hgnet, self.optimizer, opt_level='O1')
        except:
            print('[WARNING] DON`T INSTALL APEX')

        if 'cuda' == self.device:
            hgnet = nn.DataParallel(hgnet)

        return hgnet

    def load(self):
        _load_path = os.path.join(GlobalVariables.WEIGHT_DIR, self.conf.path.model.load)
        _cnt = PytorchUtil.load(self.model, _load_path, 7)

        if _cnt == 0:
            PytorchUtil.load(self.model, _load_path, 0)

    def save(self):
        pass


if __name__ == "__main__":
    from base.BaseConf import BaseConf
    import GlobalVariables

    conf = BaseConf(GlobalVariables.DEFAULT_CONF_DIR + '/HardNet.yaml')

    hardnet_model = HardNetModel(conf, None)
    time.sleep(5)
