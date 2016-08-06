import os
import importlib
import unittest

from platoon import configparser as cfgp


def test_fetch_hosts_from_envs():
    if os.getenv("PLATOON_HOSTS"):
        os.environ.pop("PLATOON_HOSTS")
    true_hosts = ["test0", "tes1", "te2"]
    os.environ["PLATOON_HOSTS"] = "test0,tes1,te2"
    importlib.reload(cfgp)
    hosts = cfgp.fetch_hosts()
    assert hosts == true_hosts, (hosts)


def test_fetch_hosts_from_rc():
    if os.getenv("PLATOON_HOSTS"):
        os.environ.pop("PLATOON_HOSTS")
    os.environ["PLATOONRC"] = "../../../platoonrc.conf"
    importlib.reload(cfgp)
    hosts = cfgp.fetch_hosts()
    assert hosts == ["lisa0", "lisa1", "lisa3"], (hosts)


def test_fetch_devices_from_envs():
    if os.getenv("PLATOON_DEVICES"):
        os.environ.pop("PLATOON_DEVICES")
    os.environ["PLATOON_DEVICES"] = "cuda0,cudae"
    importlib.reload(cfgp)
    devices = cfgp.fetch_devices_for_host("asfasfa")
    assert devices == ["cuda0", "cudae"], (devices)


def test_fetch_devices_from_rc():
    if os.getenv("PLATOON_DEVICES"):
        os.environ.pop("PLATOON_DEVICES")
    os.environ["PLATOON_DEVICES"] = ""
    os.environ["PLATOONRC"] = "../../../platoonrc.conf"
    importlib.reload(cfgp)
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
