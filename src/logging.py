import logging
from lightning.pytorch.utilities.rank_zero import rank_zero_only

class RankZeroLogger:
    def __init__(self, logger):
        self._logger = logger
        
    def debug(self, *args, **kwargs):
        return rank_zero_only(self._logger.debug)(*args, **kwargs)
        
    def info(self, *args, **kwargs):
        return rank_zero_only(self._logger.info)(*args, **kwargs)
        
    def warning(self, *args, **kwargs):
        return rank_zero_only(self._logger.warning)(*args, **kwargs)
        
    def error(self, *args, **kwargs):
        return rank_zero_only(self._logger.error)(*args, **kwargs)
    
def rank_zero_logger(logger: logging.Logger) -> RankZeroLogger:
    return RankZeroLogger(logger)
