def minimize_sum_error(a):
    """
    Given a list of non-negative integers a,
    returns a list b satisfying |b[i] - b[i+1]| <= 1 and b[i] >= 0,
    minimizing sum(|a[i] - b[i]|).
    """
    n = len(a)
    if n == 0:
        return []

    H = max(a)
    # DP table: dp[i][h] = minimum cost for prefix ending at i with b[i]=h
    dp = [[float('inf')] * (H+1) for _ in range(n)]
    # Parent pointers to reconstruct
    parent = [[None] * (H+1) for _ in range(n)]

    # Base case for i=0
    for h in range(H+1):
        dp[0][h] = abs(a[0] - h)
        parent[0][h] = -1  # start marker

    # Fill DP
    for i in range(1, n):
        for h in range(H+1):
            # possible previous heights
            for dh in (-1, 0, 1):
                ph = h + dh
                if 0 <= ph <= H:
                    cost = dp[i-1][ph] + abs(a[i] - h)
                    if cost < dp[i][h]:
                        dp[i][h] = cost
                        parent[i][h] = ph

    # Find optimal end height
    min_cost = float('inf')
    end_h = 0
    for h in range(H+1):
        if dp[n-1][h] < min_cost:
            min_cost = dp[n-1][h]
            end_h = h

    # Reconstruct b
    b = [0] * n
    h = end_h
    for i in range(n-1, -1, -1):
        b[i] = h
        h = parent[i][h]

    return b

def shape_preserving_lipschitz(a):
    n = len(a)
    b = [0] * n
    b[0] = round(a[0])  # or use average of a[0]

    for i in range(1, n):
        target_slope = a[i] - a[i-1]
        if target_slope > 0:
            b[i] = min(b[i-1] + 1, round(a[i]))
        elif target_slope < 0:
            b[i] = max(b[i-1] - 1, round(a[i]))
        else:
            b[i] = b[i-1]
        b[i] = max(b[i], 0)  # enforce non-negativity

    return b

if __name__ == '__main__':
    # Example
    a = [0, 5, 2]
    b = minimize_sum_error(a)
    c = shape_preserving_lipschitz(a)
    print("Original:", a)
    print("min-sum:", b)
    print("Sum-error:", sum(abs(ai - bi) for ai, bi in zip(a, b)))
    print("shape-lips:", c)
    print("Sum-error:", sum(abs(ai - bi) for ai, bi in zip(a, c)))