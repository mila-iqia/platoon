import six
import unittest

from platoon import controller

if six.PY3:
    buffer_ = memoryview
else:
    buffer_ = buffer  # noqa


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.local_size = 3
        cls.devices = ["cuda0", "cuda1", "cuda2"]
        cls.control = controller.Controller(5567, local_size=cls.local_size,
                                            device_list=cls.devices)

    @classmethod
    def tearDownClass(cls):
        cls.control._close()

    def test_is_worker_first(self):
        first = self.control._is_worker_first(self.control._am_i_first_count)
        assert first
        first = self.control._is_worker_first(self.control._am_i_first_count)
        assert not first
        first = self.control._is_worker_first(self.control._am_i_first_count)
        assert not first
        first = self.control._is_worker_first(self.control._am_i_first_count)
        assert first
        first = self.control._is_worker_first(self.control._am_i_first_count)
        assert not first
        first = self.control._is_worker_first(self.control._am_i_first_count)
        assert not first

    def test_get_region_info(self):
        req_info = {}

        req_info['region_id'] = bytearray(b'1')
        req_info['device'] = 'cuda0'
        res = self.control._get_region_info(req_info)
        assert set(res.keys()) == set(['region_id', 'region_size', 'regional_rank', 'multinode'])
        assert res['region_id'] == bytearray(b"platoon-1")
        assert res['region_size'] == self.local_size
        assert res['regional_rank'] == 0
        assert not res['multinode']

        req_info['region_id'] = bytearray(b'2')
        req_info['device'] = 'cuda1'
        res = self.control._get_region_info(req_info)
        assert set(res.keys()) == set(['region_id', 'region_size', 'regional_rank', 'multinode'])
        assert res['region_id'] == bytearray(b"platoon-1")
        assert res['region_size'] == self.local_size
        assert res['regional_rank'] == 1
        assert not res['multinode']

        req_info['region_id'] = bytearray(b'3')
        req_info['device'] = 'cuda2'
        res = self.control._get_region_info(req_info)
        assert set(res.keys()) == set(['region_id', 'region_size', 'regional_rank', 'multinode'])
        assert res['region_id'] == bytearray(b"platoon-1")
        assert res['region_size'] == self.local_size
        assert res['regional_rank'] == 2
        assert not res['multinode']

        req_info['region_id'] = bytearray(b'asdfasfda')
        req_info['device'] = 'cuda1'
        res = self.control._get_region_info(req_info)
        assert set(res.keys()) == set(['region_id', 'region_size', 'regional_rank', 'multinode'])
        assert res['region_id'] == bytearray(b"platoon-asdfasfda")
        assert res['region_size'] == self.local_size
        assert res['regional_rank'] == 1
        assert not res['multinode']

    def test_init_new_shmem(self):
        self.control._job_uid = "yo"
        req_info = {'size': 64}

        res = self.control._init_new_shmem(req_info)
        assert res == "platoon-yo_0_buffer"
        assert len(self.control.shared_buffers) == 1
        assert len(self.control._shmrefs) == 1
        assert self.control._last_shmem_name == "platoon-yo_0_buffer"
        a = self.control.shared_buffers[res]
        try:
            buffer_(a)
        except TypeError:
            self.fail("self.control.shared_buffers[{}] does not provide buffer interface.".format(0))
        assert len(a) == 64

        res = self.control._init_new_shmem(req_info)
        assert res == "platoon-yo_0_buffer"
        assert len(self.control.shared_buffers) == 1
        assert len(self.control._shmrefs) == 1
        assert self.control._last_shmem_name == "platoon-yo_0_buffer"
        b = self.control.shared_buffers[res]
        try:
            buffer_(b)
        except TypeError:
            self.fail("self.control.shared_buffers[{}] does not provide buffer interface.".format(0))
        assert len(b) == 64
        assert b == a

        res = self.control._init_new_shmem(req_info)
        assert res == "platoon-yo_0_buffer"
        assert len(self.control.shared_buffers) == 1
        assert len(self.control._shmrefs) == 1
        assert self.control._last_shmem_name == "platoon-yo_0_buffer"
        c = self.control.shared_buffers[res]
        try:
            buffer_(c)
        except TypeError:
            self.fail("self.control.shared_buffers[{}] does not provide buffer interface.".format(0))
        assert len(c) == 64
        assert c == a

        req_info = {'size': 512}
        res = self.control._init_new_shmem(req_info)
        assert res == "platoon-yo_1_buffer"
        assert len(self.control.shared_buffers) == 2
        assert len(self.control._shmrefs) == 2
        assert self.control._last_shmem_name == "platoon-yo_1_buffer"
        e = self.control.shared_buffers[res]
        try:
            buffer_(e)
        except TypeError:
            self.fail("self.control.shared_buffers[{}] does not provide buffer interface.".format(1))
        assert len(e) == 512
        assert e != c

        res = self.control._init_new_shmem(req_info)
        assert res == "platoon-yo_1_buffer"
        assert len(self.control.shared_buffers) == 2
        assert len(self.control._shmrefs) == 2
        assert self.control._last_shmem_name == "platoon-yo_1_buffer"
        f = self.control.shared_buffers[res]
        try:
            buffer_(f)
        except TypeError:
            self.fail("self.control.shared_buffers[{}] does not provide buffer interface.".format(1))
        assert len(f) == 512
        assert f != c
        assert f == e

        res = self.control._init_new_shmem(req_info)
        assert res == "platoon-yo_1_buffer"
        assert len(self.control.shared_buffers) == 2
        assert len(self.control._shmrefs) == 2
        assert self.control._last_shmem_name == "platoon-yo_1_buffer"
        g = self.control.shared_buffers[res]
        try:
            buffer_(g)
        except TypeError:
            self.fail("self.control.shared_buffers[{}] does not provide buffer interface.".format(1))
        assert len(g) == 512
        assert g != c
        assert g == e
