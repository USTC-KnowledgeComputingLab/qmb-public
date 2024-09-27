try:
    from ._version import version, __version__
except ImportError:
    version = __version__ = "0.0.0"
