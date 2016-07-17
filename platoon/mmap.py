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
