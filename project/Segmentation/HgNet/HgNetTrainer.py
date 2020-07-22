from base.ABCModelRunner import ABCModelRunner
from project.Segmentation.HgNet.Util.HgNetDataset import HgNetDataset
from project.Util import PytorchUtil, IouUtil
import torch.utils.data as data
import os
import GlobalVariables


class HgNetTrainer(ABCModelRunner):

    def run(self, x=None, y=None, val_x=None, val_y=None, *args, **kwargs):
        _weight_dir = os.path.join(GlobalVariables.WEIGHT_DIR, self.model.conf.path.model.save)

        os.makedirs(_weight_dir, exist_ok=True)

        with open(os.path.join(_weight_dir, 'conf.yaml'), 'w') as conf_file:
            conf_file.write(self.conf.text)

        PytorchUtil.train(self.getModel().model,
                          self.train_hgnet_dataloader,
                          self.valid_hgnet_dataloader,
                          self.model.loss,
                          self.model.optimizer,
                          epochs=self.conf.epochs,
                          acc_fn=IouUtil.class_acc,
                          weight_path=_weight_dir,
                          weight_nm=self.conf.name,
                          device=self.model.device)

    def load(self):
        load_path_conf = self.conf.path.data.load
        train_dataset = HgNetDataset(conf=self.conf, input_shape=self.model.conf.input,
                                     x_path=load_path_conf.train.x, y_path=load_path_conf.train.y)
        valid_dataset = HgNetDataset(conf=self.conf, input_shape=self.model.conf.input,
                                    x_path=load_path_conf.valid.x, y_path=load_path_conf.valid.y)
        self.train_hgnet_dataloader = data.DataLoader(dataset=train_dataset, batch_size=self.conf.batch_size,
                                                         num_workers=self.conf.num_workers, pin_memory=self.conf.pin_memory
                                                      )

        self.valid_hgnet_dataloader = data.DataLoader(dataset=valid_dataset, batch_size=self.conf.batch_size,
                                                      num_workers=self.conf.num_workers, pin_memory=self.conf.pin_memory
                                                      )

    def save(self):
        pass
