from attrdict import AttrDict
import yaml


class BaseConf:
    def __new__(cls, path: str, *args, **kwargs):
        conf_txt = open(path)
        _yaml = yaml.full_load(conf_txt)

        for key in _yaml.keys():
            _yaml[key]['text'] = open(path, 'r').read()
            
        attr_dict = AttrDict(_yaml)
        
        return attr_dict
