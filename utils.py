


def rebin2d(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0],
             new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).sum(-1).sum(1)

def rebin3d(arr, new_shape):
    shape = (-1,new_shape[-2], arr.shape[-2] // new_shape[0],
             new_shape[-1], arr.shape[-1] // new_shape[1])
    return arr.reshape(shape).sum(-1).sum(2)