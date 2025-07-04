from bisect import insort, bisect_left

def median_lower(arr):
    """Return the lower median of a list of numbers."""
    n = len(arr)
    sorted_arr = sorted(arr)
    mid = (n - 1) // 2
    return sorted_arr[mid]

class Block:
    def __init__(self, values, start, end):
        self.values = values  # list of original a_i in this block
        self.start = start    # inclusive index
        self.end = end        # inclusive index
        self.value = median_lower(values)

    def merge(self, other):
        merged_values = self.values + other.values
        merged_block = Block(merged_values, self.start, other.end)
        return merged_block


def lipschitz_pav(a):
    """
    Perform L1 Lipschitz regression with slope constraint |b[i+1] - b[i]| <= 1.
    Returns b that minimizes sum(|a[i] - b[i]|) under the constraint.
    """
    n = len(a)
    if n == 0:
        return []

    # Initialize each point as its own block
    blocks = [Block([a[i]], i, i) for i in range(n)]

    # Merge adjacent violating blocks until no violations remain
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(blocks) - 1:
            b1, b2 = blocks[i], blocks[i+1]
            if b2.value - b1.value > 1 or b1.value - b2.value > 1:
                # Violation: merge these two blocks
                new_block = b1.merge(b2)
                blocks[i:i+2] = [new_block]
                changed = True
                # After merging, restart check from previous block if exists
                i = max(i-1, 0)
            else:
                i += 1

    # Construct the fitted sequence b
    b = [0] * n
    for block in blocks:
        for idx in range(block.start, block.end + 1):
            b[idx] = block.value
    return b

if __name__ == '__main__':
    examples = [
        [0, 5, 2],
        [3, 0, 8, 4, 2],
        [5, 5, 5, 5],
        [10, 0, 10],
    ]
    for a in examples:
        b = lipschitz_pav(a)
        print(f"a = {a}\nb = {b}\nsum-error = {sum(abs(x-y) for x,y in zip(a,b))}\n")
