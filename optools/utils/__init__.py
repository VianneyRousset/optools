def clip(value, minv=None, maxv=None):
    if minv is not None:
        value = max(value, minv)

    if maxv is not None:
        value = min(value, maxv)

    return value
