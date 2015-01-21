def identity(*args):
    # return argument object(s) itself/themselves
    if len(args) == 1:
        return args[0]
    else:
        return args
