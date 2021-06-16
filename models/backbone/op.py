def _make_divisible(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor
