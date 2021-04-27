import os
import signal
import subprocess
from sys import platform

from pypads import logger
from pypads.app.env import LoggerEnv
from pypads.app.injections.run_loggers import RunSetup, RunTeardown
from pypads.utils.logging_util import get_temp_folder


class STrace(RunSetup):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _call(self, *args, _pypads_env: LoggerEnv, **kwargs):
        pads = _pypads_env.pypads

        file = os.path.join(get_temp_folder(), str(os.getpid()) + "_trace.txt")
        proc = None
        if platform == "linux" or platform == "linux2":
            # https://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
            proc = subprocess.Popen(['sudo strace -p ' + str(os.getpid()) + ' &> ' + file], shell=True,
                                    preexec_fn=os.setsid)

        elif platform == "darwin":
            proc = subprocess.Popen(['sudo dtruss -f -p ' + str(os.getpid()) + ' 2> ' + file], shell=True,
                                    preexec_fn=os.setsid)

        elif platform == "win32":
            logger.warning("No tracing supported on windows currently.")

        if proc:
            pads.api.register_teardown("stop_dtrace_" + str(proc.pid),
                                       STraceStop(_pypads_proc=proc, _pypads_trace_file=file))
            if proc.poll() == 1:
                logger.warning(
                    "Can't dtruss/strace without sudo rights. To enable tracking allow user to execute dtruss/strace "
                    "without sudo password with polkit or by modifiying visudo - /etc/sudoers:"
                    "username ALL=NOPASSWD: /usr/bin/dtruss. To get the path to dtruss you can use 'which dtruss'. "
                    "Be carefull about allowing permanent sudo rights to dtruss. This might introduce security risks.")

        def safety_hook():
            """
            A None value indicates that the process hasn't terminated yet.
            """
            if proc and proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.terminate()

        pads.add_exit_fn(safety_hook)


class STraceStop(RunTeardown):

    def __init__(self, *args, _pypads_proc=None, _pypads_trace_file=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._proc = _pypads_proc
        self._trace_file = _pypads_trace_file

    def _call(self, *args, _pypads_env: LoggerEnv, **kwargs):
        pads = _pypads_env.pypads
        if self._proc and self._proc.poll() is None:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            self._proc.terminate()
        try:
            pads.api._log_artifact(self._trace_file, description="Strace of the experiment process.")
        except Exception as e:
            pass
