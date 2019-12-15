import logging
__all__ = ['Logger']


root_logger = logging.getLogger('taurex')
root_logger.propagate = False
"""Root logger for taurex"""


class TauRexHandler(logging.StreamHandler):
    """
    Logging Handler for Taurex 3. Prevents other
    MPI threads from writing to log unless they are in trouble (>=ERROR)

    Parameters
    ----------
    stream : stream-object , optional
        Stream to write to otherwise defaults to ``stderr``

    """

    def __init__(self, stream=None):
        from taurex.mpi import get_rank
        super().__init__(stream=stream)

        self._rank = get_rank()

    def emit(self, record):
        # print(record)
        if self._rank == 0 or record.levelno >= logging.ERROR:
            # msg = '[{}] {}'.format(self._rank,record.msg)
            # record.msg = msg
            return super(TauRexHandler, self).emit(record)
        else:
            pass


rh = TauRexHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
rh.setFormatter(formatter)
rh.setLevel(logging.DEBUG)
root_logger.handlers = []
root_logger.addHandler(rh)
root_logger.setLevel(logging.INFO)


class Logger:
    """
    Standard logging using logger library

    Parameters
    -----------
    name : str
        Name used for logging

    """

    def __init__(self, name):
        self._log_name = 'taurex.{}'.format(name)

        self._logger = logging.getLogger('taurex.{}'.format(name))

    def info(self, message, *args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.info(message, *args, **kwargs)

    def warning(self, message, *args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.warning(message, *args, **kwargs)

    def debug(self, message, *args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.debug(message, *args, **kwargs)

    def error(self, message, *args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.error(message, *args, **kwargs)

    def critical(self, message, *args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.critical(message, *args, **kwargs)
