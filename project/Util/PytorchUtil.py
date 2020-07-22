from typing import Tuple
from time import time
from tqdm import tqdm
from torch import nn
import numpy as np
import torch
import os
from base.ABCModel import ABCModel
import datetime
import csv

use_apex = True
try:
    from apex import amp
except:
    use_apex = False


def load(model: ABCModel, path: str = '', slice_idx: int = 0, device='cpu', optimizer = None, **kwargs):
    if not isinstance(model, nn.Module):
        print('[ERROR] predict : model isn\'t torch nn.Module')
        raise TypeError

    load_state_dict = torch.load(path, map_location=device)

    model_state_dict = model.state_dict()

    set_state_dict = {}

    layer_count = 0

    for k, v in load_state_dict['model'].items():
        key = k[slice_idx:]
        if key in model_state_dict:
            set_state_dict[key] = v
            layer_count += 1

    model_state_dict.update(set_state_dict)
    model.load_state_dict(model_state_dict)

    if optimizer is not None:
        optimizer.load_state_dict(load_state_dict['optimizer'])

    if use_apex:
        amp.load_state_dict(load_state_dict['amp'])

    print("[LOAD] success at {} (layer count: {})".format(path, layer_count))

    return layer_count


def predict(model, data, device='cpu') -> np.ndarray:
    if not isinstance(model, nn.Module):
        print('[E0RROR] predict : model isn\'t torch nn.Module')
        raise TypeError

    if not isinstance(data, torch.Tensor):
        print('[ERROR] data isn\'t torch tensor')

    model.eval()

    with torch.no_grad():
        model = model.to(device)
        data_tensor = data.to(device, dtype=torch.float)

        _pred_result = model(data_tensor)

    return _pred_result.cpu().numpy()


def validate(model, data_loader, loss_fn, acc_fn=None, device='cpu') -> Tuple[float, float]:
    if not isinstance(model, nn.Module):
        print('[ERROR] validate : model isn\'t torch nn.Module')
        raise TypeError

    model.eval()

    _valid_loss = 0.0
    _acc_n = 0

    if acc_fn is not None:
        _acc_sum = 0.0
    else:
        _acc_sum = None

    with torch.no_grad():
        model = model.to(device)

        print("[VALID] start")
        for _data, _label in tqdm(data_loader):
            _data_tensor = _data.to(device, dtype=torch.float)
            _label_tensor = _label.to(device, dtype=torch.long)

            _pred_result = model(_data_tensor).to(device)

            _valid_loss += loss_fn(_pred_result, _label_tensor).item()

            if acc_fn is not None:
                _acc_sum += acc_fn(_pred_result, _label_tensor, device).item()
                _acc_n += 1

    return _valid_loss / _acc_n, _acc_sum / _acc_n


def train(model, train_data_loader, valid_data_loader=None, loss_fn=None, optimizer=None, epochs: int = 1, acc_fn=None,
          weight_path: str = 'weight/', weight_nm: str = 'model', device: str = 'cpu', callbacks: list = [], *args,
          **kwargs):
    if not isinstance(model, nn.Module):
        print('[ERROR] validate : model isn\'t torch nn.Module')
        raise TypeError

    os.makedirs(weight_path, exist_ok=True)

    csv_nm = 'conf.csv'

    with open(os.path.join(weight_path, csv_nm), 'a+', newline='') as conf_csv:
        conf_writer = csv.writer(conf_csv)
        conf_writer.writerow(['epoch', 'loss', acc_fn.__name__, 'val_loss', 'val_{}'.format(acc_fn.__name__)])

    model = model.to(device)
    model.train()

    best_val_acc = 0.

    for epoch in range(epochs):
        _train_acc_sum = 0.0

        _train_loss_sum = 0.0
        _train_loss_n = 0

        print("[TRAIN] start")

        for i, (_train_data, _train_label) in enumerate(train_data_loader):
            _start_time = time()
            _train_data_tensor = _train_data.to(device, dtype=torch.float)
            _train_label_tensor = _train_label.to(device, dtype=torch.long)

            optimizer.zero_grad()

            _train_result = model(_train_data_tensor)

            _loss = loss_fn(_train_result, _train_label_tensor)

            if use_apex:
                with amp.scale_loss(_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                _loss.backward()

            optimizer.step()

            _train_loss_sum += _loss.item()
            _train_loss_n += 1

            _train_acc_sum += acc_fn(_train_result, _train_label_tensor, device)

            _batch_time = time() - _start_time
            _remain_time = (len(train_data_loader) - i) * _batch_time

            print('{}/{} {:.4f}% ETA: {} loss: {:.4f} {}: {:.4f}        '.format(i,
                                                                                 len(train_data_loader),
                                                                                 (i / len(train_data_loader)) * 100,
                                                                                 str(datetime.timedelta(
                                                                                     seconds=int(_remain_time))),
                                                                                 _train_loss_sum / _train_loss_n,
                                                                                 acc_fn.__name__,
                                                                                 _train_acc_sum / _train_loss_n),
                  end='\r')

        print(
            '[train] epoch {}: {}                                     '.format(epoch, _train_loss_sum / _train_loss_n))

        _val_acc = None
        _valid_loss = None

        if valid_data_loader:
            _valid_loss, _val_acc = validate(model, valid_data_loader, loss_fn, acc_fn, device)
            print('[valid] epoch {}: {}'.format(epoch, _valid_loss))
            model.train()

        with open(os.path.join(weight_path, csv_nm), 'a+', newline='') as conf_csv:
            conf_writer = csv.writer(conf_csv)
            conf_writer.writerow(
                [epoch, _train_loss_sum / _train_loss_n, _train_acc_sum.item() / _train_loss_n, _valid_loss, _val_acc])

        if _val_acc is not None:
            # check best weight
            if _val_acc < best_val_acc:
                continue

            best_val_acc = _val_acc

            # save weight
            _weight_nm = os.path.join(weight_path,
                                      '{}_{}_{}_acc_{:.4f}.pt'.format(weight_nm, int(time()), epoch, _val_acc))

        else:
            _weight_nm = os.path.join(weight_path, '{}_{}_{}_val_{}_{:.4f}.pt'.format(weight_nm,
                                                                                      int(time()),
                                                                                      epoch,
                                                                                      acc_fn.__name__,
                                                                                      _train_acc_sum / _train_loss_n))

        save_dict = {'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'amp': amp.state_dict() if use_apex else None}
        torch.save(save_dict, _weight_nm)
        print('[SAVE] weight save at {}'.format(_weight_nm))

    conf_csv.close()
    # return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list
