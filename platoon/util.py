from __future__ import print_function
import os
import subprocess
import cffi


def mmap(length=0, prot=0x3, flags=0x1, fd=0, offset=0):
    _ffi = cffi.FFI()
    _ffi.cdef("void *mmap(void *, size_t, int, int, int, size_t);")
    _lib = _ffi.dlopen(None)

    addr = _ffi.NULL

    m = _lib.mmap(addr, length, prot, flags, fd, offset)
    if m == _ffi.cast('void *', -1):
        raise OSError(_ffi.errno, "for mmap")
    return _ffi.buffer(m, length)


def launch_process(logs_folder, experiment_name, args, device, process_type="worker"):
    print("Starting {0} on {1} ...".format(process_type, device), end=' ')

    log_file = os.path.join(logs_folder, "{0}{1}.{{}}".format(process_type, device))
    with open(log_file.format("out"), 'w') as stdout_file:
        with open(log_file.format("err"), 'w') as stderr_file:
            env = dict(os.environ)
            env['THEANO_FLAGS'] = '{},device={}'.format(env.get('THEANO_FLAGS', ''), device)
            command = ["python", "-u", "{0}_{1}.py".format(experiment_name, process_type)]
            if args is not None:
                command += args
            process = subprocess.Popen(command, bufsize=0, stdout=stdout_file, stderr=stderr_file, env=env)

    print("Done")
    return process
