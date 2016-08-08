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
        cls.control = controller.Controller(5567, devices=cls.devices)

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

    def test_get_platoon_info(self):
        req_info = {}

        req_info['local_id'] = '1'
        req_info['device'] = 'cuda0'
        res = self.control._get_platoon_info(req_info)
        assert set(res.keys()) == set(['local_id', 'local_size', 'local_rank', 'multinode', 'global_size'])
        assert res['local_id'] == "platoon-1"
        assert res['local_size'] == self.local_size
        assert res['local_rank'] == 0
        assert not res['multinode']
        assert res['global_size'] == self.local_size

        req_info['local_id'] = '2'
        req_info['device'] = 'cuda1'
        res = self.control._get_platoon_info(req_info)
        assert set(res.keys()) == set(['local_id', 'local_size', 'local_rank', 'multinode', 'global_size'])
        assert res['local_id'] == "platoon-1"
        assert res['local_size'] == self.local_size
        assert res['local_rank'] == 1
        assert not res['multinode']
        assert res['global_size'] == self.local_size

        req_info['local_id'] = '3'
        req_info['device'] = 'cuda2'
        res = self.control._get_platoon_info(req_info)
        assert set(res.keys()) == set(['local_id', 'local_size', 'local_rank', 'multinode', 'global_size'])
        assert res['local_id'] == "platoon-1"
        assert res['local_size'] == self.local_size
        assert res['local_rank'] == 2
        assert not res['multinode']
        assert res['global_size'] == self.local_size

        req_info['local_id'] = 'asdfasfda'
        req_info['device'] = 'cuda1'
        res = self.control._get_platoon_info(req_info)
        assert set(res.keys()) == set(['local_id', 'local_size', 'local_rank', 'multinode', 'global_size'])
        assert res['local_id'] == "platoon-asdfasfda"
        assert res['local_size'] == self.local_size
        assert res['local_rank'] == 1
        assert not res['multinode']
        assert res['global_size'] == self.local_size

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
