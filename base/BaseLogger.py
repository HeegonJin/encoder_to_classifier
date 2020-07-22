from logging.handlers import TimedRotatingFileHandler
from attrdict import AttrDict
from GlobalVariables import *
import logging
import os


class BaseLogger:
    def __init__(self, conf):
        self.logger_dict = self.getLogger(conf)
        default_logger = self.getDefaultLogger(conf.project.name)
        if default_logger is not None and DEFAULT_LOGGER not in self.logger_dict:
            self.logger_dict[DEFAULT_LOGGER] = default_logger

    @classmethod
    def getLogger(cls, conf: AttrDict) -> dict:
        logger_dict = {}

        for _logger in conf.project.logger:
            _logger_category = list(_logger.keys())[0]
            _logger_dict = _logger(_logger_category)

            logger = cls._makeLogger(conf.project.name, _logger_category, _logger_dict)
            logger_dict[_logger_category] = logger
        return logger_dict

    @classmethod
    def getDefaultLogger(cls, project_name: str) -> logging.Logger:
        default_dict = AttrDict({'path': DEFAULT_LOGGER_PATH, 'level': 'debug'})
        logger = cls._makeLogger(project_name, DEFAULT_LOGGER, default_dict)
        return logger

    @classmethod
    def _makeLogger(cls, project_name: str, logger_category: str, logger_dict: AttrDict) -> logging.Logger:
        logger = logging.getLogger(logger_category)

        if len(logger.handlers) > 0:
            return None

        log_level = cls.getLevel(logger_dict.level)
        logger.setLevel(log_level)

        formatter = logging.Formatter('[{}]'.format(project_name) +
                                      '[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] : %(message)s')
        stream_hander = logging.StreamHandler()
        stream_hander.setFormatter(formatter)
        logger.addHandler(stream_hander)

        if 'path' in logger_dict:
            _log_path = logger_dict.path
        else:
            print('[WARNING] Don`t set logger path')
            _log_path = DEFAULT_LOGGER_PATH

        _log_dir = os.path.dirname(_log_path)
        if not os.path.exists(_log_dir):
            os.makedirs(_log_dir)

        file_handler = TimedRotatingFileHandler(_log_path, encoding='utf-8', when='midnight')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    @classmethod
    def getLevel(cls, level_str: str):
        if level_str.lower() == 'notset':
            return logging.NOTSET
        elif level_str.lower() == 'debug':
            return logging.DEBUG
        elif level_str.lower() == 'info':
            return logging.INFO
        elif level_str.lower() == 'warning':
            return logging.WARNING
        elif level_str.lower() == 'error':
            return logging.ERROR
        elif level_str.lower() == 'critical':
            return logging.CRITICAL
        else:
            raise TypeError(level_str + ' is not match.')

    def debug(self, category: str, msg: str, *args):
        logger = self._getCacheLogger(category)
        logger.debug(msg.format(*args))

    def info(self, category: str, msg: str, *args):
        logger = self._getCacheLogger(category)
        logger.info(msg.format(*args))

    def warning(self, category: str, msg: str, *args):
        logger = self._getCacheLogger(category)
        logger.warning(msg.format(*args))

    def error(self, category: str, msg: str, *args):
        logger = self._getCacheLogger(category)
        logger.error(msg.format(*args))

    def critical(self, category: str, msg: str, *args):
        logger = self._getCacheLogger(category)
        logger.critical(msg.format(*args))

    def _getCacheLogger(self, category):
        return self.logger_dict[category] if category in self.logger_dict else self.logger_dict[DEFAULT_LOGGER]
