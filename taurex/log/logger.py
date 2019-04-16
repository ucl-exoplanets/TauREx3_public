import logging
import threading
from multiprocessing import Queue
import multiprocessing
__all__ = ['Logger','ProcessLogger']

class TauRexLogger(object):
    """Base class for logging in TauRex

    This class wraps logging functionality and provides them to the derived classes
    providing info,debug,critical, error and warning methods

    Should not be used directly

    Parameters
    -----------
    name : str
        Name used for logging

    """
    _proc_log_queue = Queue()

    

    _init = False
    @classmethod
    def getLogQueue(cls):
        """Provides logging queue for multiprocessing logging
        
        Returns
        --------
        :obj:`multiprocessing.Queue`
            Queue where logs should go
        
        """
        return cls._proc_log_queue

    @classmethod
    def _logging_thread(cls):
        """This thread collects logs from Processes and writes them to stream"""

        thread_log = TauRexLogger.getLogger('log_thread')
        log_queue = cls.getLogQueue()

        thread_log.info('Starting Multiprocess logging')
        while True:
            
            name,log_level,message,args,kwargs = log_queue.get()
            _log = logging.getLogger(name)
            _log.log(log_level,message,*args,**kwargs)

    @classmethod
    def getRootLogger(cls):
        return cls._root_logger   
    
    @classmethod
    def getLogger(cls,name):
        return logging.getLogger('taurex.{}'.format(name))

    @classmethod
    def reInit(cls):
        if cls._init is False:
            cls._root_logger = logging.getLogger('taurex')


            cls._root_logger.info('Reinitializing taurexLogger')
            cls._log_thread = threading.Thread(target=cls._logging_thread) 
            cls._log_thread.daemon = True
            cls._log_thread.start()
        cls._init = True
        


    def __init__(self,name):
        self._log_name = 'taurex.{}'.format(name)
        TauRexLogger.reInit()

    @property
    def logName(self):
        return self._log_name

    def info(self,message,*args, **kwargs):
        pass
    def warning(self,message,*args, **kwargs):
        pass
    def debug(self,message,*args, **kwargs):
        pass
    
    def error(self,message,*args, **kwargs):
        pass
    
    def critical(self,message,*args, **kwargs):
        pass

class Logger(TauRexLogger):
    """Standard logging using logger library

    Parameters
    -----------
    name : str
        Name used for logging

    """    

    def __init__(self,name):
        TauRexLogger.__init__(self,name)
        self._logger = logging.getLogger(self.logName)

    def info(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.info(message,*args,**kwargs)
    def warning(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.warning(message,*args,**kwargs)
    def debug(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.debug(message,*args,**kwargs)
    
    def error(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.error(message,*args,**kwargs)
    
    def critical(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._logger.critical(message,*args,**kwargs)


class ProcessLogger(TauRexLogger):
    """Sends logs to queue to be processed by logging thread

    Parameters
    -----------
    name : str
        Name used for logging

    """    

    def __init__(self,name):
        TauRexLogger.__init__(self,name)
        self._logger = logging.getLogger(self.logName)
        self._log_queue = TauRexLogger.getLogQueue()
    def info(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._log_queue.put( (self._log_name,logging.INFO,message,args,kwargs) )
    def warning(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._log_queue.put( (self._log_name,logging.WARNING,message,args,kwargs) )
    def debug(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._log_queue.put( (self._log_name,logging.DEBUG,message,args,kwargs) )
    
    def error(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._log_queue.put( (self._log_name,logging.ERROR,message,args,kwargs) )
    
    def critical(self,message,*args, **kwargs):
        """ See :class:`logging.Logger` """
        self._log_queue.put( (self._log_name,logging.CRITICAL,message,args,kwargs) )




def main():
    pass
if __name__=="__main__":
    main()