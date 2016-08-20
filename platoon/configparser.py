import os
import shlex
import sys

from six.moves import configparser as ConfigParser

# The PLATOON_DEVICES environment variable should be a list of comma-separated
# device name entries, e.g. PLATOON_DEVICES=cuda0,cuda2,cuda3
PLATOON_DEVICES = os.getenv("PLATOON_DEVICES", "")

# The PLATOON_HOSTS environment variable should be a list of comma-separated
# host machine entries, e.g. PLATOON_HOSTS=lisa1,ceylon
PLATOON_HOSTS = os.getenv("PLATOON_HOSTS", "")


def config_files_from_platoonrc():
    if sys.platform != "win32":
        rval = [os.path.expanduser('~/.platoonrc')]
        rval.append(os.path.join(os.getcwd(), '.platoonrc'))
    else:
        rval = [os.path.expanduser('~/.platoonrc.txt')]
        rval.append(os.path.join(os.getcwd(), '.platoonrc.txt'))
    if os.getenv('PLATOONRC') is not None:
        rval.extend([os.path.expanduser(s) for s in
                     os.getenv('PLATOONRC').split(os.pathsep)])
    return rval

config_files = config_files_from_platoonrc()
platoon_cfg = ConfigParser.SafeConfigParser(
    {'USER': os.getenv("USER", os.path.split(os.path.expanduser('~'))[-1]),
     'LSCRATCH': os.getenv("LSCRATCH", ""),
     'TMPDIR': os.getenv("TMPDIR", ""),
     'TEMP': os.getenv("TEMP", ""),
     'TMP': os.getenv("TMP", ""),
     'PID': str(os.getpid()),
     }
)
platoon_cfg.optionxform = str
platoon_cfg.read(config_files)
# Having a raw version of the config around as well enables us to pass
# through config values that contain format strings.
# The time required to parse the config twice is negligible.
platoon_raw_cfg = ConfigParser.RawConfigParser()
platoon_raw_cfg.optionxform = str
platoon_raw_cfg.read(config_files)


def fetch_devices_for_host(host):
    """A successful search returns a list of theano devices' string values.
    An unsuccessful search raises a KeyError.

    The (decreasing) priority order is:
    - PLATOON_DEVICES
    - PLATOONRC files (if they exist) from right to left
    - working directory's ./.platoonrc
    - ~/.platoonrc

    """
    # first try to have PLATOON_DEVICES
    if PLATOON_DEVICES:
        splitter = shlex.shlex(PLATOON_DEVICES, posix=True)
        splitter.whitespace += ','
        splitter.whitespace_split = True
        return list(splitter)

    # next try to find it in the config file
    try:
        try:
            devices = platoon_cfg.get("devices", host)
        except ConfigParser.InterpolationError:
            devices = platoon_raw_cfg.get("devices", host)
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        raise KeyError(host)
    splitter = shlex.shlex(devices, posix=True)
    splitter.whitespace += ','
    splitter.whitespace_split = True
    return list(splitter)


def fetch_hosts():
    """A successful search returns a list of host to participate in a multi-node
    platoon. An unsuccessful search raises a KeyError.

    The (decreasing) priority order is:
    - PLATOON_HOSTS
    - PLATOONRC files (if they exist) from right to left
    - working directory's ./.platoonrc
    - ~/.platoonrc

    """
    # first try to have PLATOON_HOSTS
    if PLATOON_HOSTS:
        splitter = shlex.shlex(PLATOON_HOSTS, posix=True)
        splitter.whitespace += ','
        splitter.whitespace_split = True
        return list(splitter)

    # next try to find it in the config file
    try:
        try:
            hosts = platoon_cfg.get("platoon", "hosts")
        except ConfigParser.InterpolationError:
            hosts = platoon_raw_cfg.get("platoon", "hosts")
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError):
        raise KeyError("hosts")
    splitter = shlex.shlex(hosts, posix=True)
    splitter.whitespace += ','
    splitter.whitespace_split = True
    return list(splitter)
