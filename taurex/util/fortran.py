import multiprocessing as mp
from queue import Empty, Full
from .module import getattr_recursive, setattr_recursive, runfunc_recursive
import importlib
from contextlib import contextmanager
import io
import os
import sys
import tempfile

@contextmanager
def stdout_redirector(stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    try:
        original_stdout_fd = sys.stdout.fileno()
    except io.UnsupportedOperation:
        yield
        return

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        #libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        if stream is not None:
            stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


class FortranStopException(Exception):
    pass


class StreamToLogQueue:

    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            out = line.rstrip()

            if isinstance(out, bytes):
                out = out.decode('utf-8')
            try:
                self.log_queue.put_nowait(out)
            except Full:
                pass



class SafeFortranProcess(mp.Process):

    def __init__(self, work_queue, output_queue, shutdown_event,
                 module_string, log_queue=None):
        super().__init__()
        self._module_string = module_string
        self.work_queue = work_queue
        self.output_queue = output_queue
        self.shutdown_event = shutdown_event
        self.log_queue = log_queue

    def run(self):
        timeout = 1.0

        mod = importlib.import_module(self._module_string)
        if self.log_queue is not None:
            logger = StreamToLogQueue(self.log_queue)

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
                    with stdout_redirector(logger):
                        out = 'Success', runfunc_recursive(mod, attr, *args, **kwargs)
                except Exception as e:
                    out = 'Exception', str(e)

            self.output_queue.put(out)


class SafeFortranCaller:

    STOP_WAIT_SECS = 2.0

    def __init__(self, module, logger=None):

        self._module = module
        self._process = None
        self._message_queue = None
        self._output_queue = None
        self._log_queue = None
        self.shutdown_event = None
        self.logger = logger

    @property
    def process_queues(self):
        if self._process is None:
            self.cleanup()
            self._message_queue = mp.Queue()
            self.shutdown_event = mp.Event()
            self._output_queue = mp.Queue()
            self._log_queue = mp.Queue()
            self._process = SafeFortranProcess(self._message_queue,
                                               self._output_queue,
                                               self.shutdown_event,
                                               self._module, self._log_queue)
            self._process.daemon = True
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
        m.put(('CALL', attr, args, kwargs))

        while p.is_alive():
            self.process_logs()
            try:
                success, value = o.get(block=True, timeout=timeout)
                break
            except Empty:
                continue
        
        self.process_logs()

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

    def process_logs(self):
        try:
            item = self._log_queue.get(block=False)
        except Empty:
            item = None

        while item:
            if self.logger is not None:
                self.logger.info(item)
            try:
                item = self._log_queue.get(block=False)
            except Empty:
                break

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
        if self._log_queue is not None:
            self._cleanup_queue(self._log_queue)
            self._log_queue = None

    def cleanup(self):
        self.cleanup_processes()
        try:
            self.cleanup_queues()
        except OSError:
            pass

    def __del__(self):
        self.cleanup()
