from abc import ABC

import torch.distributed


class Serializer(ABC):
    def __init__():
        super().__init__()

        self.__sync_index = 0
        self.__store = None

    def get_sync_index(self):
        if torch.distributed.is_available():
            if self.__store is None:
                self.__store = torch.distributed.distributed_c10d._get_default_store()

            idx = self.__store.add(f"{type(self).__name__}_sync_index") - 1
            return idx

        idx = self.__sync_index
        self.__sync_index += 1

        return idx
