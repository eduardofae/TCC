import pkgutil
import importlib

def import_all_submodules(package_name):
    package = importlib.import_module(package_name)
    for loader, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        importlib.import_module(name)