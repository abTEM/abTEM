from numba import jit, prange


@jit(nopython=True, parallel=True, nogil=True)
def window_and_collapse(S, window, corners, coefficients):
    N, M = S.shape[1:]
    n, m = window.shape[1:]
    for k in prange(len(corners)):
        i, j = corners[k]
        ti = n - (N - i)
        tj = m - (M - j)

        if (i + n <= N) & (j + m <= M):
            for l in range(len(coefficients[k])):
                window[k, :] = S[l, i:i + n, j:j + m] * coefficients[k][l]

        elif (i + n <= N) & (j + m > M):
            for l in range(len(coefficients[k])):
                window[k, :, :-tj] += S[l, i:i + n, j:] * coefficients[k][l]
                window[k, :, -tj:] += S[l, i:i + n, :tj] * coefficients[k][l]

        if (i + n > N) & (j + m <= M):
            for l in range(len(coefficients[k])):
                window[k, :-ti, :] += S[l, i:, j:j + m] * coefficients[k][l]
                window[k, -ti:, :] += S[l, :ti, j:j + m] * coefficients[k][l]

        elif (i + n > N) & (j + m > M):
            for l in range(len(coefficients[k])):
                window[k, :-ti, :-tj] += S[l, i:, j:] * coefficients[k][l]
                window[k, :-ti, -tj:] += S[l, i:, :tj] * coefficients[k][l]
                window[k, -ti:, -tj:] += S[l, :ti, :tj] * coefficients[k][l]
                window[k, -ti:, :-tj] += S[l, :ti, j:] * coefficients[k][l]
