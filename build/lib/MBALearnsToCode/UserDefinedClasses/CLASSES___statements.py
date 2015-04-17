from copy import deepcopy


class Statement:
    def __init__(self, statement, translations___dict={}):
        self.statement = statement
        self.translations = translations___dict

    def eval(self, dict_object):
        d = rename_dict_keys(dict_object, self.translations)
        return eval(self.statement, d)


def rename_dict_keys(dict_object, to_new_keys_from_old_keys___dict):
    d = deepcopy(dict_object)   # just to be careful #
    for new_key, old_key in to_new_keys_from_old_keys___dict.items():
        value = d[old_key]
        del d[old_key]
        d[new_key] = value
    return d