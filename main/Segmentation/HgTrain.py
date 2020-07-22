from project.Segmentation.HgNet.HgNetTrainer import HgNetTrainer
from project.Segmentation.HgNet.HgNetModel import HgNetModel
from base.BaseConf import BaseConf
import GlobalVariables
import os

if __name__ == "__main__":
    _conf_path = os.path.join(GlobalVariables.CONF_SEG_DIR, 'SegTrainerConf.yaml')
    conf = BaseConf(_conf_path)

    hgnet_model = HgNetModel(conf.SegModel, None)
    hgnet_trainer = HgNetTrainer(model=hgnet_model, conf=conf.SegTrainer, logger=None)
    hgnet_trainer.load()
    hgnet_trainer.run()
    hgnet_trainer.save()
