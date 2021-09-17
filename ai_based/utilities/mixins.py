class DictLike:
    """This class makes any inheritor's members accessible via dictionary-like [] accesses."""
    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self.__getattribute__(item)
