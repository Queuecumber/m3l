import inspect
from pkgutil import iter_modules

import hydra_configs.torch
from hydra.core.config_store import ConfigStore


def register_hydra():
    search_modules = [("hydra_configs.torch", hydra_configs.torch)]
    classes = set()

    while search_modules:
        p, m = search_modules.pop()
        for name, obj in inspect.getmembers(m):
            if inspect.isclass(obj) and name.endswith("Conf"):
                reg_name = name.replace("Conf", "").lower()

                path = obj.__module__.replace("hydra_configs.torch.", "")
                path = reversed(path.split("."))

                group = ""
                for mod in path:
                    if mod.lower() != reg_name:
                        group = mod
                        break

                classes.add((group, reg_name, obj))

        if "__path__" in m.__dict__:
            for submodule in iter_modules(m.__path__, prefix=f"{p}."):
                c = __import__(submodule.name, fromlist="dummy")
                search_modules.append((submodule.name, c))

    cs = ConfigStore.instance()
    for group, name, node in classes:
        cs.store(group=group, name=name, node=node)
