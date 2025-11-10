from typing import Final, Iterable, Mapping, Optional, Tuple, Union

import paddle

Key = Union[str, Tuple[str, ...]]


class ModuleDict(paddle.nn.LayerDict):
    CLASS_ATTRS: Final[Tuple[str, ...]] = tuple(dir(paddle.nn.LayerDict))

    def __init__(
        self,
        modules: Optional[Mapping[Union[str, Tuple[str, ...]],
                                  paddle.nn.Layer]] = None,
    ):
        if modules is not None:
            modules = {
                self.to_internal_key(key): module
                for key, module in modules.items()
            }
        super().__init__(modules)

    @classmethod
    def to_internal_key(cls, key: Key) -> str:
        if isinstance(key, tuple):
            assert len(key) > 1
            key = f"<{'___'.join(key)}>"
        assert isinstance(key, str)
        if key in cls.CLASS_ATTRS:
            key = f"<{key}>"
        return key.replace(".", "#")

    @classmethod
    def to_external_key(cls, key: str) -> Key:
        key = key.replace("#", ".")
        if key[0] == "<" and key[-1] == ">" and key[1:-1] in cls.CLASS_ATTRS:
            key = key[1:-1]
        if key[0] == "<" and key[-1] == ">" and "___" in key:
            key = tuple(key[1:-1].split("___"))
        return key

    def __getitem__(self, key: Key) -> paddle.nn.Layer:
        return super().__getitem__(self.to_internal_key(key))

    def __setitem__(self, key: Key, module: paddle.nn.Layer):
        return super().__setitem__(self.to_internal_key(key), module)

    def __delitem__(self, key: Key):
        return super().__delitem__(self.to_internal_key(key))

    def __contains__(self, key: Key) -> bool:
        return super().__contains__(self.to_internal_key(key))

    def keys(self) -> Iterable[Key]:
        return [self.to_external_key(key) for key in super().keys()]

    def items(self) -> Iterable[Tuple[Key, paddle.nn.Layer]]:
        return [(self.to_external_key(k), v) for k, v in super().items()]
