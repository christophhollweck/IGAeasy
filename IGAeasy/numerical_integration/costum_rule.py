import numpy as np

def gauss_lobatto(n):
    """
    Gauss-Lobatto points and weights on [-1, 1] for n = 2 to 5.
    Returns (points, weights)
    """
    import numpy as np
    if n == 2:
        x = np.array([-1.0, 1.0])
        w = np.array([1.0, 1.0])
    elif n == 3:
        x = np.array([-1.0, 0.0, 1.0])
        w = np.array([1/3, 4/3, 1/3])
    elif n == 4:
        x = np.array([-1.0, -np.sqrt(1/5), np.sqrt(1/5), 1.0])
        w = np.array([1/6, 5/6, 5/6, 1/6])
    elif n == 5:
        x = np.array([-1.0, -np.sqrt(3/7), 0.0, np.sqrt(3/7), 1.0])
        w = np.array([1/10, 49/90, 32/45, 49/90, 1/10])
    else:
        raise NotImplementedError("Only n=2 to 5 implemented.")
    return [x, w]
