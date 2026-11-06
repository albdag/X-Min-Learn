"""
Handles application's exceptions and crashes logging.

Created on Tue Nov 03 11:34:21 2026
@author: albdag
"""

import faulthandler
import gzip
import logging
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import TracebackType

from PyQt5.QtWidgets import QApplication


class CrashLogger:

    def __init__(self, root_dir: Path) -> None:
        '''
        A class for tracking exceptions and application crashes globally. Full
        exception tracebacks are logged and stored on dedicated log files.

        Parameters
        ----------
        root_dir : Path
            Path where the logs folder is located.

        '''
        # Create a new latest.log file
        self._logs_dir = self._get_logs_dir(root_dir)
        self._path = self._logs_dir / 'latest.log'

        # Archive previous latest.log file
        self._archive_previous_log()

        # Initialize and configure a new logger instance
        self._logger = self._init_logger()

        # Enable faulthandler to catch hard OS-Level crashes
        self._fault_file = open(self._path, "a", encoding="utf-8") # noqa: SIM115
        faulthandler.enable(file=self._fault_file)


    @property
    def log_filepath(self) -> str:
        '''
        A string representation of the log filepath.

        Returns
        -------
        str
            Path to log file.

        '''
        return str(self._path)


    @property
    def log_location_msg(self) -> str:
        '''
        Standard instruction message pointing to the active log file location.

        Returns
        -------
        str
            Log location message.

        '''
        return f'For more details, check log file at: {self.log_filepath}'


    def _get_logs_dir(self, root: Path) -> Path:
        '''
        Get the full path to logs folder. If not existing, the logs folder is
        created.

        Parameters
        ----------
        root : Path
            Path where the logs folder should be found or created.

        Returns
        -------
        Path
            Path to logs folder.

        '''
        logs_dir = root / 'logs'
        if not logs_dir.is_dir():
            logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir


    def _archive_previous_log(self) -> None:
        '''
        Compress existing latest.log file into a timestamped .gz archive if 
        non-empty.

        '''
        # Exit function if a previous latest.log file does not exist
        if not self._path.exists():
            return

        # Use file modification time for the archive name
        mtime = datetime.fromtimestamp(self._path.stat().st_mtime)
        timestamp_str = mtime.strftime("%Y%m%d_%H%M%S")
        archive_path = self._logs_dir / f"archived_log_{timestamp_str}.log.gz"

        # Compress old latest.log if it has content
        if self._path.stat().st_size > 0:
            try:
                with (
                    open(self._path, 'rb') as f_in,
                    gzip.open(archive_path, 'wb') as f_out
                ):
                    shutil.copyfileobj(f_in, f_out)
            except Exception: # noqa: BLE001, S110
                pass

        # Remove old uncompressed latest.log
        try:
            self._path.unlink()
        except OSError:
            pass


    def _init_logger(self) -> logging.Logger:
        '''
        Intialize a new logger instance and configure it with a FileHandler.

        Returns
        -------
        logger : logging.Logger
            New logger instance.

        '''
        logger = logging.getLogger("XMinLearnLogger")
        logger.setLevel(logging.WARNING)
        
        # Attach a new FileHandler pointing to the timestamped crash log file
        handler = logging.FileHandler(self._path, encoding='utf-8')
        formatter = logging.Formatter(
            fmt='%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        
        # Clear existing handlers to prevent writing to old session log files
        logger.handlers.clear()
        logger.addHandler(handler)
            
        return logger


    def log_exception(
        self,
        e: BaseException,
        context_msg: str = '',
        level: int = logging.WARNING
    ) -> str:
        '''
        Log an exception traceback to the log file with logging level 'level'
        and return the full traceback string.

        Parameters
        ----------
        e : BaseException
            Exception to be added to log.
        context_msg : str, optional
            A description of the context when the exception raised. The default
            is ''.
        level : int
            Logging level. Valid levels are logging.ERROR and logging.WARNING.
            The default is logging.WARNING.

        Returns
        -------
        tb_str : str
            Full traceback string.

        '''
        tb_str = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        full_msg = f"{context_msg}\n{tb_str}" if context_msg else tb_str
        self._logger.log(level, full_msg)
        return tb_str


    def handle_uncaught_exception(self, e: BaseException) -> None:
        '''
        Deal with uncaught exception 'e' that caused the application to crash.

        Parameters
        ----------
        e : BaseException
            Uncaught exception.

        '''
        # Log exception with ERROR level
        msg = 'UNCAUGHT EXCEPTION THAT CAUSED APP CRASH'
        full_traceback = self.log_exception(e, msg, logging.ERROR)

        # Print full traceback to console (only useful for development)
        print(full_traceback) 
        
        # Flush faulthandler log buffer
        self._fault_file.flush()


    def handle_caught_exception(
            self, e: BaseException, c: object | type | None = None) -> None:
        '''
        Deal with caught exception 'e' that was handled by the application.

        Parameters
        ----------
        e : BaseException
            Caught exception.
        c : object | type | None, optional
            Class instance or class type whose method raised the exception. If 
            provided, the class name will be appended to the context message in
            log file. The default is None.

        '''
        # Generate context message
        msg = 'Caught exception'
        if c is not None:
            name = c.__name__ if isinstance(c, type) else c.__class__.__name__
            msg += f' raised in class {name}'

        # Log exception as with WARNING level (the default)
        self.log_exception(e, msg)


    def open_log(self) -> None:
        '''
        Open the log file. This works cross-platform.

        '''
        if not self._path.exists():
            return

        match sys.platform:
            case 'win32':
                os.startfile(self.log_filepath)
            case 'darwin':
                subprocess.run(["open", self.log_filepath], check=False)
            case _:
                # Check if xdg-open exists on Linux/Unix systems
                if shutil.which('xdg-open'): 
                    subprocess.run(['xdg-open', self.log_filepath], check=False)
                # Fallback: attempt opening editor via $EDITOR env var
                else:
                    editor = os.environ.get('EDITOR', 'nano')
                    subprocess.run([editor, self.log_filepath], check=False)


    def close(self) -> None:
        '''
        Disable faulthandler and cleanly close the log file stream. This method
        is meant to be called only on normal application exit (see 'main.py').

        '''
        faulthandler.disable()
        if not self._fault_file.closed:
            self._fault_file.flush()
            self._fault_file.close()



logger = None
def setup_logger(root: Path) -> None:
    '''
    Setup the application crash logger by providing the path to the 'root'
    directory where logs folder should be found or created. This function is 
    meant to be called only during application initialization (see 'main.py').

    Parameters
    ----------
    root : Path
        Path to directory where logs folder is stored.

    '''
    global logger
    logger = CrashLogger(root)


def custom_excepthook(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback: TracebackType | None
) -> None:
    '''
    Global exception hook to intercept unhandled crashes, log full tracebacks,
    and display an informative crash dialog before exit. This function is meant
    to be called very early during application initialization (see 'main.py').

    Parameters
    ----------
    exc_type : type[BaseException]
        The exception class type.
    exc_value : BaseException
        The exception instance.
    exc_traceback : TracebackType | None
        The exception traceback.

    '''
    # Skip keybord interruption exception
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # Exit function if crash logger has not been set (safety)
    if logger is None:
        return 

    # Log uncaught exception to crash log
    logger.handle_uncaught_exception(exc_value)

    # Display crash dialog and terminate app cleanly
    _ = QApplication.instance() or QApplication(sys.argv) # ensure app exists
    from dialogs import CrashDialog
    dialog = CrashDialog(logger)
    dialog.exec_()
    sys.exit(1)