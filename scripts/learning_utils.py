"""
Just some extra code to learn things.
"""

def binary_search_min(x_min, x_max, loss_fn, eps_frac=1e-4):
    """
    Find the value within [x_min, x_max] that minimizes loss_fn

    loss_fun should be bitonic up (roughly convex)

    Iterates until the estimate converges, which happens when updates
    change the estimate by less than (x_max-x_min)*eps_frac
    """

    if x_min > x_max:
        raise ValueError('x_min should be less than x_max')

    eps = (x_max - x_min) * eps_frac / 2.0

    while True:
        x_cur = (x_min + x_max) / 2.0
        value = loss_fn(x_cur)
        v_less_dx = loss_fn(x_cur - eps)
        v_plus_dx = loss_fn(x_cur + eps)
        if v_less_dx > value and value > v_plus_dx:
            x_min = x_cur
        elif v_less_dx < value and value < v_plus_dx:
            x_max = x_cur
        elif v_less_dx < value and value > v_plus_dx:
            raise ValueError('loss fn not bitonic up')
        else:
            # v_less_dx > value and value < v_plus_dx:
            # x_cur minimizes the loss function at the granularity
            # of epsilon, so we can exit
            break

    return (x_cur, value)

def test1(x):
    return (x+4)**2

x, v = binary_search_min(-10, 10, test1)
print 'Found min at x={0}, v={1}'.format(x, v)