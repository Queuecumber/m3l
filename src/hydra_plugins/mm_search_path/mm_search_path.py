from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin
from os import environ


class MMSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend("mm", f"file:///etc/mm")
        search_path.prepend("mm", f"file://{environ['HOME']}/.config/mm")