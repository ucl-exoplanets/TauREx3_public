from .logger import Logger
import logging

last_log = logging.INFO


def setLogLevel(level):
    global last_log
    from .logger import root_logger
    root_logger.setLevel(level)
    last_log = level


def disableLogging():
    import logging
    from .logger import root_logger
    root_logger.setLevel(logging.ERROR)


def enableLogging():
    global last_log
    import logging
    if last_log is None:
        last_log = logging.INFO
    setLogLevel(last_log)
