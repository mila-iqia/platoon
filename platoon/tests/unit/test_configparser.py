from __future__ import absolute_import
import os
import unittest
from six.moves import reload_module as reload

from ... import configparser as cfgp


def test_fetch_hosts_from_envs():
    if os.getenv("PLATOON_HOSTS"):
        os.environ.pop("PLATOON_HOSTS")
    true_hosts = ["test0", "tes1", "te2"]
    os.environ["PLATOON_HOSTS"] = "test0,tes1,te2"
    reload(cfgp)
    hosts = cfgp.fetch_hosts()
    assert hosts == true_hosts, (hosts)


def test_fetch_hosts_from_rc():
    if os.getenv("PLATOON_HOSTS"):
        os.environ.pop("PLATOON_HOSTS")
    os.environ["PLATOONRC"] = "../../../platoonrc.conf"
    reload(cfgp)
    hosts = cfgp.fetch_hosts()
    assert hosts == ["lisa0", "lisa1", "lisa3"], (hosts)


def test_fetch_devices_from_envs():
    if os.getenv("PLATOON_DEVICES"):
        os.environ.pop("PLATOON_DEVICES")
    os.environ["PLATOON_DEVICES"] = "cuda0,opencl0:1"
    reload(cfgp)
    devices = cfgp.fetch_devices_for_host("asfasfa")
    assert devices == ["cuda0", "opencl0:1"], (devices)


def test_fetch_devices_from_rc():
    if os.getenv("PLATOON_DEVICES"):
        os.environ.pop("PLATOON_DEVICES")
    os.environ["PLATOON_DEVICES"] = ""
    os.environ["PLATOONRC"] = "../../../platoonrc.conf"
    reload(cfgp)
    devs = cfgp.fetch_devices_for_host("lisa0")
    assert devs == ["cuda0", "cuda1"], (devs)
    devs = cfgp.fetch_devices_for_host("lisa1")
    assert devs == ["cuda3", "cuda0"], (devs)
    devs = cfgp.fetch_devices_for_host("lisa3")
    assert devs == ["cuda"], (devs)
    keyerror = False
    try:
        devs = cfgp.fetch_devices_for_host("asfasfa")
    except KeyError:
        keyerror = True
    except:
        pass
    assert keyerror
