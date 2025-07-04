from typing import List


def minimize_sum_shape(a: List[int]) -> List[int]:
    """
    Minimizes the sum of absolute errors |a_i - b_i| under the constraint |b[i+1] - b[i]| <= 1,
    while favoring the shape (sign of slopes) of the original sequence.

    This dynamic-programming algorithm uses a lexicographic objective:
      1) Minimize total error.
      2) Maximize number of matching slope-signs with the original data.

    Returns the fitted sequence b with the same length as a.
    """
    n = len(a)
    if n == 0:
        return []

    H = max(a)
    # Precompute original slope signs: +1, -1, or 0
    orig_sign = [0] * n
    for i in range(1, n):
        if a[i] > a[i-1]: orig_sign[i] = 1
        elif a[i] < a[i-1]: orig_sign[i] = -1
        else: orig_sign[i] = 0

    # DP tables
    INF = 10**15
    dp_error = [[INF] * (H+1) for _ in range(n)]
    dp_score = [[-INF] * (H+1) for _ in range(n)]
    parent = [[None] * (H+1) for _ in range(n)]

    # Base case
    for h in range(H+1):
        dp_error[0][h] = abs(a[0] - h)
        dp_score[0][h] = 0

    # Fill DP
    for i in range(1, n):
        for h in range(H+1):
            for dh in (-1, 0, 1):
                ph = h + dh
                if 0 <= ph <= H:
                    prev_e = dp_error[i-1][ph]
                    if prev_e == INF:
                        continue
                    # New error
                    e = prev_e + abs(a[i] - h)
                    # Compute slope sign match
                    if h > ph:
                        sign = 1
                    elif h < ph:
                        sign = -1
                    else:
                        sign = 0
                    match = 1 if sign == orig_sign[i] and sign != 0 else 0
                    s = dp_score[i-1][ph] + match
                    # Lexicographic comparison: prefer lower error, then higher score
                    if (e < dp_error[i][h]) or (e == dp_error[i][h] and s > dp_score[i][h]):
                        dp_error[i][h] = e
                        dp_score[i][h] = s
                        parent[i][h] = ph

    # Choose best ending height
    best_h = 0
    for h in range(1, H+1):
        if (dp_error[n-1][h] < dp_error[n-1][best_h]) or \
           (dp_error[n-1][h] == dp_error[n-1][best_h] and dp_score[n-1][h] > dp_score[n-1][best_h]):
            best_h = h

    # Reconstruct solution
    b = [0] * n
    h = best_h
    for i in range(n-1, -1, -1):
        b[i] = h
        h = parent[i][h]

    return b


if __name__ == '__main__':
    examples = [
        [0, 5, 2],
        [3, 0, 8, 4, 2],
        [5, 5, 5, 5],
        [10, 0, 10],
    ]
    for a in examples:
        b = minimize_sum_shape(a)
        print(f"a = {a}\nb = {b}\nsum-error = {sum(abs(x-y) for x,y in zip(a,b))}")
