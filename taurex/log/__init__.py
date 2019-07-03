from .logger import Logger




def setLogLevel(level):
    
    from .logger import root_logger
    root_logger.setLevel(level)


def disableLogging():
    import logging
    setLogLevel(logging.ERROR)