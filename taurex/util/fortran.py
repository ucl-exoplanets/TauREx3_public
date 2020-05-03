import multiprocessing as mp
from queue import Empty
from .module import getattr_recursive, setattr_recursive, runfunc_recursive
import importlib


class FortranStopException(Exception):
    pass


class SafeFortranProcess(mp.Process):

    def __init__(self, work_queue, output_queue, shutdown_event,
                 module_string):
        super().__init__()
        self._module_string = module_string
        self.work_queue = work_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event

    def run(self):
        timeout = 1.0

        mod = importlib.import_module(self._module_string)

        while not self.shutdown_event.is_set():
            try:
                item = self.work_queue.get(block=True, timeout=timeout)
            except Empty:
                continue

            if item == 'END':
                break

            task = item[0]
            attr = item[1]
            out = None, None
            if task == 'GET':
                try:
                    out = 'Success', getattr_recursive(mod, attr)
                except Exception as e:
                    out = 'Exception', e

            elif task == 'SET':
                value = item[2]
                try:
                    out = 'Success', setattr_recursive(mod, attr, value)
                except Exception as e:
                    out = 'Exception', e
            elif task == 'CALL':
                args = item[2]
                kwargs = item[3]
                try:
                    out = 'Success', runfunc_recursive(mod, attr, *args, 
                                                      **kwargs)
                except Exception as e:
                    out = 'Exception', str(e)

            self.output_queue.put(out)


class SafeFortranCaller:

    STOP_WAIT_SECS = 2.0

    def __init__(self, module):

        self._module = module
        self._process = None
        self._message_queue = None
        self._output_queue = None
        self.shutdown_event = None

    @property
    def process_queues(self):
        if self._process is None:
            self.cleanup()
            self._message_queue = mp.Queue()
            self.shutdown_event = mp.Event()
            self._output_queue = mp.Queue()
            self._process = SafeFortranProcess(self._message_queue,
                                               self._output_queue,
                                               self.shutdown_event,
                                               self._module)
            self._process.start()
        return self._process, self._message_queue, self._output_queue

    def get_val(self, attr):
        p, m, o = self.process_queues

        success, value = 'Crash', None
        timeout = 1.0
        m.put(('GET', attr))

        while p.is_alive():
            try:
                success, value = o.get(block=True, timeout=timeout)
                break
            except Empty:
                continue
        if success == 'Crash':
            self.cleanup()
            raise FortranStopException
        elif success == 'Exception':
            self.cleanup()
            raise Exception(value)
        elif success == 'Success':
            return value
        else:
            self.cleanup()
            raise Exception(f'Unknown exception {attr}-{success}-{value}')

    def set_val(self, attr, set_value):
        p, m, o = self.process_queues

        success, value = 'Crash', None
        timeout = 1.0
        m.put(('SET', attr, set_value))

        while p.is_alive():
            try:
                success, value = o.get(block=True, timeout=timeout)
                break
            except Empty:
                continue
        if success == 'Crash':
            self.cleanup()
            raise FortranStopException
        elif success == 'Exception':
            self.cleanup()
            raise Exception(value)
        elif success == 'Success':
            return value
        else:
            self.cleanup()
            raise Exception(f'Unknown exception {attr}-{set_value}-'
                            f'{success}-{value}')

    def call(self, attr, *args, **kwargs):
        p, m, o = self.process_queues

        success, value = 'Crash', None
        timeout = 1.0
        m.put(('CA::', attr, args, kwargs))

        while p.is_alive():
            try:
                success, value = o.get(block=True, timeout=timeout)
                break
            except Empty:
                continue
        if success == 'Crash':
            self.cleanup()
            raise FortranStopException
        elif success == 'Exception':
            self.cleanup()
            raise Exception(value)
        elif success == 'Success':
            return value
        else:
            self.cleanup()
            raise Exception(f'Unknown exception {attr}'
                            f'-{success}-{value}')

    def cleanup_processes(self):

        if self.shutdown_event is not None:
            self.shutdown_event.set()

        end_time = self.STOP_WAIT_SECS

        if self._process is not None:

            self._process.join(end_time)
            if self._process.is_alive():
                self._process.terminate()
        self._process = None
        self.shutdown_event = None

    def _cleanup_queue(self, queue):
        try:
            item = queue.get(block=False)
        except Empty:
            item = None

        while item:
            try:
                item = queue.get(block=False)
            except Empty:
                break

        queue.close()
        queue.join_thread()

    def cleanup_queues(self):
        if self._message_queue is not None:
            self._cleanup_queue(self._message_queue)
            self._message_queue = None
        if self._output_queue is not None:
            self._cleanup_queue(self._output_queue)
            self._output_queue = None

    def cleanup(self):
        self.cleanup_processes()
        self.cleanup_queues()

    def __del__(self):
        self.cleanup()
