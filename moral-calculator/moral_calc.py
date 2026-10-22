"""Moral Calculator: a pure-Python port of the MATLAB Moral Calculator app.

For a 2x2 material game and a total moral weight W, this computes which subgame outcomes
(A,A), (A,B), (B,A), (B,B) can be sustained as a subgame perfect equilibrium of the moral game
at each point of an (inequality, individualism) grid, as in "An Endogenous Moral Formation Game".

FIDELITY (what this file replicates, and how faithfully)
-------------------------------------------------------
Source ported: ~/papers/morality/moral_app_20260403.m (the April 2026 app build), the functions
reached from the app's "Generate Graph" button:
    weightOutcomes -> generateWeights -> supportedOutcomes
        -> findFeasibleCornersFullSpace -> convhull -> generatePerturbedTestPoints
        -> profDeviation -> getUtilGivenMoral, getBoundsForCDEF, possibleMoralDev
Each Python function below names its MATLAB original. The port keeps every tolerance
(1e-10, 1e-14, 1e-8, 2e-12, 1e-7) and the floating-point evaluation order of every expression,
so results are meant to be bit-identical, not merely close.

MATLAB primitives (checked bit-for-bit against MATLAB R2025a by verify/check_primitives.py):
  - linspace: MATLAB's own formula, not numpy's. Identical on all 9 probe ranges.
  - unique(...,'rows'): lexicographic sort, duplicates removed. Same semantics.
  - convhull: same SET semantics (every boundary point kept, including points on an edge; an
    all-collinear input raises, and the app then keeps every point), exact orientation tests.
  - norm of a 2-vector: BLAS scaled formula, identical on 3999/4000 probe vectors.
  - 2^x: NOT reproducible. MATLAB's own math library differs from every Python candidate by
    +-1 ulp on ~10% of inputs (no candidate formula matched). The weights w_ij can therefore
    differ from MATLAB's in the last bit; verify/ shows which grid cells, if any, this flips.

Not ported (unreachable from the app's button): findFeasibleCorners, sortPointsForPerimeter,
linspace2D, generatePointsAlongEdges. The nFS argument is accepted by MATLAB but unused.

Known difference from replication_code.m (5 Apr 2026, used for the paper's figures): that
file's generatePerturbedTestPoints does NOT clip test points to [-1,1]^2; the app does, and so
does this port. verify/ measures whether the clip ever changes an answer.

Notation: the MATLAB code (and so this port) writes every moral weight with its indices
transposed relative to the paper. The paper's w_ij (how much player i cares about player j's
value judgment) is the code's w_ji: player 1's payoff here is u1 + w11*alpha1 + w21*alpha2.
The values agree with the paper's formulas (verify/check_notation.py); only the names differ.

No dependencies beyond the Python standard library (so it also runs in a browser via Pyodide).
"""

import math
from fractions import Fraction

__version__ = "1.0"

NAN = float("nan")
INF = math.inf
OUTCOME_CODES = {1: "(A,A)", 2: "(A,B)", 4: "(B,A)", 8: "(B,B)"}


# =============================================================================
# MATLAB primitives
# =============================================================================
def matlab_linspace(d1, d2, n):
    """MATLAB R2025a linspace.m, line for line (numpy's formula differs in the last bit)."""
    n = math.floor(float(n))
    if n <= 0:
        return []
    n1 = n - 1
    d1 = float(d1)
    d2 = float(d2)
    if d1 == -d2 and n > 2:
        step = d2 / n1
        y = [k * step for k in range(-n1, n1 + 1, 2)]
        y[0] = d1
        y[-1] = d2
        if n1 % 2 == 0:
            y[n1 // 2] = 0.0
        return y
    if n1 == 0:  # MATLAB: 0/0 = NaN, then y(1) = d2
        return [d2]
    c = (d2 - d1) * (n1 - 1)
    if math.isinf(c):
        if math.isinf(d2 - d1):
            y = [d1 + (d2 / n1) * k - (d1 / n1) * k for k in range(n1 + 1)]
        else:
            y = [d1 + k * ((d2 - d1) / n1) for k in range(n1 + 1)]
    else:
        y = [d1 + (k * (d2 - d1)) / n1 for k in range(n1 + 1)]
    if d1 == d2:
        y = [d1] * len(y)
    else:
        y[0] = d1
        y[-1] = d2
    return y


def matlab_pow2(x):
    """MATLAB 2^x for a double scalar (overflow gives Inf, as in MATLAB)."""
    try:
        return 2.0 ** x
    except OverflowError:
        return INF


def matlab_norm2(x, y):
    """MATLAB norm([x; y]): the scaled BLAS dnrm2 formula.

    Matches MATLAB R2025a bit-for-bit on 3999 of 4000 probe vectors (math.hypot: 2636/4000).
    """
    s = max(abs(x), abs(y))
    if s == 0:
        return 0.0
    return s * math.sqrt((x / s) ** 2 + (y / s) ** 2)


def unique_rows(points):
    """MATLAB unique(P','rows')': lexicographically sorted, duplicates removed."""
    out = []
    for p in sorted(points):
        if not out or out[-1] != p:
            out.append(p)
    return out


class DegenerateHull(Exception):
    """MATLAB:convhull:EmptyConvhull2DErrId (all points collinear)."""


def _orient(a, b, c):
    """Sign of the orientation of (a, b, c), exact (Shewchuk filter, then rational arithmetic)."""
    detleft = (a[0] - c[0]) * (b[1] - c[1])
    detright = (a[1] - c[1]) * (b[0] - c[0])
    det = detleft - detright
    errbound = 3.3306690738754716e-16 * (abs(detleft) + abs(detright))
    if det > errbound:
        return 1
    if -det > errbound:
        return -1
    F = Fraction
    d = (F(a[0]) - F(c[0])) * (F(b[1]) - F(c[1])) - (F(a[1]) - F(c[1])) * (F(b[0]) - F(c[0]))
    return (d > 0) - (d < 0)


def matlab_convhull_points(points):
    """The SET of points MATLAB's convhull(x,y) returns (default 'Simplify' false).

    MATLAB returns every input point on the hull boundary, including points lying on an edge,
    and raises if all points are collinear. Only the set matters here: the caller passes the
    result through unique(), which discards the hull's ordering.
    """
    pts = unique_rows(points)
    lower = []
    for p in pts:
        while len(lower) >= 2 and _orient(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _orient(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        raise DegenerateHull
    edges = [(hull[i], hull[(i + 1) % len(hull)]) for i in range(len(hull))]
    out = []
    for p in pts:
        for a, b in edges:
            if (min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
                    and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
                    and _orient(a, b, p) == 0):
                out.append(p)
                break
    return out


def _finite(x, y):
    return not (math.isnan(x) or math.isinf(x) or math.isnan(y) or math.isinf(y))


# =============================================================================
# findFeasibleCornersFullSpace  (moral_app_20260403.m lines 315-478)
# =============================================================================
def find_feasible_corners_full_space(f1, d1, c2, d2, w11, w12, w21, w22, epsPert):
    delta1_min, delta1_max = -1.0, 1.0
    delta2_min, delta2_max = -1.0, 1.0
    epsVal = 1e-10
    cand = []

    def add(x, y):
        if _finite(x, y):
            cand.append((x, y))

    add(delta1_min, delta2_min)
    add(delta1_min, delta2_max)
    add(delta1_max, delta2_min)
    add(delta1_max, delta2_max)
    add(0.0, delta2_min)
    add(0.0, delta2_max)
    add(delta1_min, 0.0)
    add(delta1_max, 0.0)
    add(0.0, 0.0)
    if abs(w21) > epsVal:
        add(delta1_min, (f1 - d1 - w11 * delta1_min) / w21)
        add(delta1_max, (f1 - d1 - w11 * delta1_max) / w21)
    if abs(w22) > epsVal:
        add(delta1_min, (d2 - c2 - w12 * delta1_min) / w22)
        add(delta1_max, (d2 - c2 - w12 * delta1_max) / w22)
    if abs(w11) > epsVal:
        add((f1 - d1 - w21 * delta2_min) / w11, delta2_min)
        add((f1 - d1 - w21 * delta2_max) / w11, delta2_max)
    if abs(w12) > epsVal:
        add((d2 - c2 - w22 * delta2_min) / w12, delta2_min)
        add((d2 - c2 - w22 * delta2_max) / w12, delta2_max)
    if abs(w21) > epsVal:
        add(0.0, (f1 - d1 - w11 * 0) / w21)
    if abs(w22) > epsVal:
        add(0.0, (d2 - c2 - w12 * 0) / w22)
    if abs(w11) > epsVal:
        add((f1 - d1 - w21 * 0) / w11, 0.0)
    if abs(w12) > epsVal:
        add((d2 - c2 - w22 * 0) / w12, 0.0)
    if abs(w21) > epsVal and abs(w22) > epsVal:
        A = w22 * (-w11) - w21 * (-w12)
        B = w22 * (f1 - d1) - w21 * (d2 - c2)
        if abs(A) > epsVal:
            delta1_int = B / A
            delta2_int = (f1 - d1 - w11 * delta1_int) / w21
            add(delta1_int, delta2_int)

    if cand:
        cand = unique_rows(cand)

    feasible = []
    for x_, y_ in cand:
        inBox = (x_ >= delta1_min - epsVal and x_ <= delta1_max + epsVal
                 and y_ >= delta2_min - epsVal and y_ <= delta2_max + epsVal)
        if not inBox:
            continue
        lhsMin, lhsMax = -INF, INF
        if abs(w21) > epsVal:
            lhsMin = (f1 - d1 - w11 * x_) / w21
        if abs(w22) > epsVal:
            lhsMax = (d2 - c2 - w12 * x_) / w22
        if y_ >= lhsMin - epsVal and y_ <= lhsMax + epsVal:
            feasible.append((x_, y_))

    if not feasible:
        return []

    quadrants = ((-1.0, 0.0, -1.0, 0.0), (-1.0, 0.0, 0.0, 1.0),
                 (0.0, 1.0, -1.0, 0.0), (0.0, 1.0, 0.0, 1.0))
    allTest = []
    for q1min, q1max, q2min, q2max in quadrants:
        qc = [(x_, y_) for x_, y_ in feasible
              if x_ >= q1min - epsVal and x_ <= q1max + epsVal
              and y_ >= q2min - epsVal and y_ <= q2max + epsVal]
        if len(qc) < 3:
            allTest.extend(qc)
            continue
        try:
            allTest.extend(matlab_convhull_points(qc))
        except DegenerateHull:  # MATLAB: catch -> hullCorners = qCorners
            allTest.extend(qc)

    if allTest:
        allTest = unique_rows(allTest)
    if allTest and epsPert > 0:
        allTest = generate_perturbed_test_points(allTest, epsPert, epsVal)
    return allTest


# =============================================================================
# generatePerturbedTestPoints  (lines 487-598)
# NB: hullCorners arrives lexicographically sorted (unique() above), so "previous" and
# "next" corner mean neighbours in that sorted order. Replicated as is.
# =============================================================================
def generate_perturbed_test_points(hullCorners, epsPert, epsVal):
    n = len(hullCorners)
    if n < 2:
        return list(hullCorners)
    tp = []
    for i in range(n):
        cx, cy = hullCorners[i]
        px, py = hullCorners[(i - 1) % n]
        nx, ny = hullCorners[(i + 1) % n]
        tp.append((cx, cy))

        dpx, dpy = px - cx, py - cy
        normPrev = matlab_norm2(dpx, dpy)
        if normPrev > epsVal:
            tp.append((cx + (epsPert * dpx) / normPrev, cy + (epsPert * dpy) / normPrev))
        dnx, dny = nx - cx, ny - cy
        normNext = matlab_norm2(dnx, dny)
        if normNext > epsVal:
            tp.append((cx + (epsPert * dnx) / normNext, cy + (epsPert * dny) / normNext))

        onD1 = abs(cx) < epsVal
        onD2 = abs(cy) < epsVal
        if onD1 and onD2:
            continue
        elif onD1:
            tp.append((cx + -epsPert, cy + 0.0))
            tp.append((cx + epsPert, cy + 0.0))
        elif onD2:
            tp.append((cx + 0.0, cy + -epsPert))
            tp.append((cx + 0.0, cy + epsPert))

        ex, ey = cx, cy          # edge_i
        fx, fy = nx, ny          # edge_j
        if (ex < -epsVal and fx > epsVal) or (ex > epsVal and fx < -epsVal):
            t = -ex / (fx - ex)
            kx = ex + t * (fx - ex)
            ky = ey + t * (fy - ey)
            if abs(kx) > epsVal:
                kx = 0.0
            tp.append((kx, ky))
            # both branches of MATLAB's "if crossPoint(2) > 0" are identical
            if ex < 0:
                tp.append((kx + -epsPert, ky + 0.0))
            else:
                tp.append((kx + epsPert, ky + 0.0))
            ax, ay = fx - ex, fy - ey
            normEdge = matlab_norm2(ax, ay)
            if normEdge > epsVal:
                tp.append((kx + (epsPert * ax) / normEdge, ky + (epsPert * ay) / normEdge))
                tp.append((kx - (epsPert * ax) / normEdge, ky - (epsPert * ay) / normEdge))

        if (ey < -epsVal and fy > epsVal) or (ey > epsVal and fy < -epsVal):
            t = -ey / (fy - ey)
            kx = ex + t * (fx - ex)
            ky = ey + t * (fy - ey)
            if abs(ky) > epsVal:
                ky = 0.0
            tp.append((kx, ky))
            if ey < 0:
                tp.append((kx + 0.0, ky + -epsPert))
            else:
                tp.append((kx + 0.0, ky + epsPert))
            ax, ay = fx - ex, fy - ey
            normEdge = matlab_norm2(ax, ay)
            if normEdge > epsVal:
                tp.append((kx + (epsPert * ax) / normEdge, ky + (epsPert * ay) / normEdge))
                tp.append((kx - (epsPert * ax) / normEdge, ky - (epsPert * ay) / normEdge))

    # clip to [-1,1]^2 (app only; replication_code.m omits this) and unique
    tp = [(max(-1.0, min(1.0, x)), max(-1.0, min(1.0, y))) for x, y in tp]
    return unique_rows(tp)


# =============================================================================
# getBoundsForCDEF and helpers  (lines 729-913)
# =============================================================================
def _single_ineq_to_interval(A, B):
    """singleIneq2Interval: solve A + B*x >= 0."""
    if abs(B) < 1e-14:
        if A >= 0:
            return -INF, INF
        return INF, -INF
    bound = -A / B
    if B > 0:
        return bound, INF
    return -INF, bound


def _interval(A1, B1, A2, B2):
    """getIntervalForFreeDelta1 / getIntervalForFreeDelta2 (identical code)."""
    low1, high1 = _single_ineq_to_interval(A1, B1)
    low2, high2 = _single_ineq_to_interval(A2, B2)
    lowVal = max(low1, low2)
    highVal = min(high1, high2)
    if lowVal > highVal:
        return NAN, NAN
    return lowVal, highVal


def get_bounds_for_cdef(c1, d1, e1, f1, c2, d2, e2, f2, w11, w12, w21, w22, index, deltaFixed):
    if index == 1:
        k = deltaFixed[1]
        cl, ch = _interval((c1 - e1) + w21 * k, w11, (c2 - d2) + w22 * k, w12)
        dl, dh = _interval((d1 - f1) + w21 * k, w11, (d2 - c2) - w22 * k, -w12)
        el, eh = _interval((e1 - c1) - w21 * k, -w11, (e2 - f2) + w22 * k, w12)
        fl, fh = _interval((f1 - d1) - w21 * k, -w11, (f2 - e2) - w22 * k, -w12)
    elif index == 2:
        k = deltaFixed[0]
        cl, ch = _interval((c1 - e1) + w11 * k, w21, (c2 - d2) + w12 * k, w22)
        dl, dh = _interval((d1 - f1) + w11 * k, w21, (d2 - c2) - w12 * k, w22)
        el, eh = _interval((e1 - c1) - w11 * k, -w21, (e2 - f2) + w12 * k, w22)
        fl, fh = _interval((f1 - d1) - w11 * k, -w21, (f2 - e2) - w12 * k, -w22)
    else:
        raise ValueError("index must be 1 or 2.")
    return (cl, ch, dl, dh, el, eh, fl, fh)


# =============================================================================
# getUtilGivenMoral  (lines 922-1011)
# Returns (u1c, u1d, u1e, u1f, u2c, u2d, u2e, u2f); NaN where the outcome is not a subgame NE.
# =============================================================================
def get_util_given_moral(c1, d1, e1, f1, c2, d2, e2, f2, w11, w12, w21, w22, delta):
    delta1, delta2 = delta
    NE_tol = 1e-10
    cNE = (c1 - e1 + w11 * delta1 + w21 * delta2 >= -NE_tol and
           c2 - d2 + w12 * delta1 + w22 * delta2 >= -NE_tol)
    dNE = (d1 - f1 + w11 * delta1 + w21 * delta2 >= -NE_tol and
           d2 - c2 - w12 * delta1 - w22 * delta2 >= -NE_tol)
    eNE = (e1 - c1 - w11 * delta1 - w21 * delta2 >= -NE_tol and
           e2 - f2 + w12 * delta1 + w22 * delta2 >= -NE_tol)
    fNE = (f1 - d1 - w11 * delta1 - w21 * delta2 >= -NE_tol and
           f2 - e2 - w12 * delta1 - w22 * delta2 >= -NE_tol)

    if delta1 > 0:
        alpha1, beta1 = 1.0, 1 - delta1
    elif delta1 < 0:
        alpha1, beta1 = 1 + delta1, 1.0
    else:
        alpha1, beta1 = 1.0, 1.0
    if delta2 > 0:
        alpha2, beta2 = 1.0, 1 - delta2
    elif delta2 < 0:
        alpha2, beta2 = 1 + delta2, 1.0
    else:
        alpha2, beta2 = 1.0, 1.0

    u1c = u1d = u1e = u1f = u2c = u2d = u2e = u2f = NAN
    if cNE:
        u1c = c1 + w11 * alpha1 + w21 * alpha2
        u2c = c2 + w12 * alpha1 + w22 * alpha2
    if dNE:
        u1d = d1 + w11 * alpha1 + w21 * alpha2
        u2d = d2 + w12 * beta1 + w22 * beta2
    if eNE:
        u1e = e1 + w11 * beta1 + w21 * beta2
        u2e = e2 + w12 * alpha1 + w22 * alpha2
    if fNE:
        u1f = f1 + w11 * beta1 + w21 * beta2
        u2f = f2 + w12 * beta1 + w22 * beta2
    return (u1c, u1d, u1e, u1f, u2c, u2d, u2e, u2f)


# =============================================================================
# possibleMoralDev  (lines 1018-1054)
# =============================================================================
def possible_moral_dev(values):
    res = []
    for v in values:
        if v >= -1 and v <= 1:
            res.extend((v, v + 1e-8, v - 1e-8))
    if not any(abs(r + 1) < 1e-8 for r in res):
        res.append(-1.0)
    if not any(abs(r - 1) < 1e-8 for r in res):
        res.append(1.0)
    out = []
    for r in sorted(res):
        if not out or out[-1] != r:
            out.append(r)
    return [r for r in out if r >= -1 and r <= 1]


# =============================================================================
# profDeviation  (lines 1061-1178)
# outcome: 1=(A,A) 2=(A,B) 3=(B,A) 4=(B,B); player: 1 or 2
# =============================================================================
def prof_deviation(c1, d1, e1, f1, c2, d2, e2, f2, w11, w12, w21, w22,
                   outcome, moral, player, rtc):
    g = (c1, d1, e1, f1, c2, d2, e2, f2, w11, w12, w21, w22)
    row0 = 4 * (player - 1)
    current = get_util_given_moral(*g, moral)[row0 + outcome - 1]
    if math.isnan(current):
        raise RuntimeError("current utility is NaN.")
    devs = possible_moral_dev(get_bounds_for_cdef(*g, player, moral))
    other = moral[2 - player]
    for dv in devs:
        newDelta = (dv, other) if player == 1 else (other, dv)
        row = get_util_given_moral(*g, newDelta)[row0:row0 + 4]
        valid = [p for p in row if not math.isnan(p)]
        if not valid:
            return True  # empty subgame: treated as a profitable deviation
        if rtc == 1 and not math.isnan(row[outcome - 1]):
            worst = row[outcome - 1]
        else:
            worst = min(valid)
        if worst > current + 2e-12:
            return True
    return False


# =============================================================================
# supportedOutcomes  (lines 1187-1333)
# =============================================================================
def supported_outcomes(c1, d1, e1, f1, c2, d2, e2, f2, w11, w12, w21, w22, rtc):
    g = (c1, d1, e1, f1, c2, d2, e2, f2, w11, w12, w21, w22)
    ctrue = dtrue = etrue = ftrue = 0
    if c1 - e1 + w11 + w21 >= 0 and c2 - d2 + w12 + w22 >= 0:
        ctrue = 1
    if f1 - d1 + w11 + w21 >= 0 and f2 - e2 + w12 + w22 >= 0:
        ftrue = 1

    epsPert = 1e-7
    dtest = find_feasible_corners_full_space(f1, d1, c2, d2, w11, w12, w21, w22, epsPert)
    etest = find_feasible_corners_full_space(f2, e2, c1, e1, w12, w11, w22, w21, epsPert)
    if dtest:
        dtrue = 1
    if etest:
        etrue = 1

    if ctrue == 1:
        for player in (1, 2):
            if prof_deviation(*g, 1, (1.0, 1.0), player, rtc):
                ctrue = 0
                break
    if ftrue == 1:
        for player in (1, 2):
            if prof_deviation(*g, 4, (-1.0, -1.0), player, rtc):
                ftrue = 0
                break

    for flag, tests, outcome in (("d", dtest, 2), ("e", etest, 3)):
        if not tests:
            continue
        found = False
        for pt in tests:
            u = get_util_given_moral(*g, pt)
            if math.isnan(u[outcome - 1]) or math.isnan(u[4 + outcome - 1]):
                continue
            if (not prof_deviation(*g, outcome, pt, 1, rtc)
                    and not prof_deviation(*g, outcome, pt, 2, rtc)):
                found = True
                break
        if not found:
            if flag == "d":
                dtrue = 0
            else:
                etrue = 0
    return ctrue, dtrue, etrue, ftrue


# =============================================================================
# generateWeights  (lines 1345-1367)
# =============================================================================
def generate_weights(W, Ine, Ind):
    return weights_from_exp(W, matlab_pow2(Ine), matlab_pow2(Ind))


def weights_from_exp(W, expIne, expInd):
    """The rest of generateWeights, given 2^Ine and 2^Ind (lets verify/ inject MATLAB's 2^x)."""
    w11 = W * (expIne / (expIne + 1)) * (expInd / (expInd + 1))
    w12 = W * (expIne / (expIne + 1)) * (1 / (expInd + 1))
    w21 = W * (1 / (expIne + 1)) * (1 / (expInd + 1))
    w22 = W * (1 / (expIne + 1)) * (expInd / (expInd + 1))
    return w11, w12, w21, w22


# =============================================================================
# weightOutcomes for one W, as the app calls it  (lines 1376-1432, app lines 97-115)
# =============================================================================
def compute_grid(payoffs, W, ine_range=(-5.0, 5.0), ind_range=(-5.0, 5.0),
                 n_ine=100, n_ind=100, rtc=False, progress=None, pow_override=None):
    """Return (Inevec, Indvec, codes) with codes[iIne][iInd] in 0..15.

    payoffs: (u1(A,A), u2(A,A), u1(A,B), u2(A,B), u1(B,A), u2(B,A), u1(B,B), u2(B,B)),
             the order of the app's input boxes, i.e. (c1, c2, d1, d2, e1, e2, f1, f2).
    code = 1*(A,A) + 2*(A,B) + 4*(B,A) + 8*(B,B) for the outcomes supported at that point.
    pow_override: (list of 2^Ine, list of 2^Ind) to use instead of computing 2^x (verification).
    """
    c1, c2, d1, d2, e1, e2, f1, f2 = (float(v) for v in payoffs)
    Inevec = matlab_linspace(ine_range[0], ine_range[1], n_ine)
    Indvec = matlab_linspace(ind_range[0], ind_range[1], n_ind)
    if pow_override:
        powIne, powInd = pow_override
    else:
        powIne = [matlab_pow2(x) for x in Inevec]
        powInd = [matlab_pow2(x) for x in Indvec]
    rtc = 1 if rtc else 0
    W = float(W)
    codes = []
    for i, Ine in enumerate(Inevec):
        row = []
        for j, Ind in enumerate(Indvec):
            w11, w12, w21, w22 = weights_from_exp(W, powIne[i], powInd[j])
            ct, dt, et, ft = supported_outcomes(c1, d1, e1, f1, c2, d2, e2, f2,
                                                w11, w12, w21, w22, rtc)
            row.append(1 * ct + 2 * dt + 4 * et + 8 * ft)
        codes.append(row)
        if progress:
            progress(i + 1, len(Inevec))
    return Inevec, Indvec, codes


def describe_code(code):
    """'9' -> '(A,A) + (B,B)'; 0 -> 'none'."""
    parts = [name for bit, name in OUTCOME_CODES.items() if code & bit]
    return " + ".join(parts) if parts else "none"


# The app's 16-colour map (plot2DHeatmapsSeparate), code k -> colour k.
COLORS = [
    (0.9, 0.9, 0.9), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
    (1.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.5, 0.5, 0.5),
    (0.8, 0.5, 0.2), (1.0, 0.5, 0.0), (0.6, 0.0, 0.8), (0.4, 0.8, 0.4),
    (0.2, 0.6, 0.8), (0.8, 0.4, 0.2), (0.6, 0.2, 0.4), (0.2, 0.2, 0.2),
]


# =============================================================================
# Command line
# =============================================================================
def _main(argv=None):
    import argparse
    import time

    ap = argparse.ArgumentParser(
        description="Moral Calculator: supported subgame outcomes on an "
                    "(inequality, individualism) grid for a 2x2 game.")
    ap.add_argument("--payoffs", nargs=8, type=float, required=True,
                    metavar=("u1AA", "u2AA", "u1AB", "u2AB", "u1BA", "u2BA", "u1BB", "u2BB"))
    ap.add_argument("--W", type=float, default=3.0, help="total moral weight (default 3)")
    ap.add_argument("--ine", nargs=2, type=float, default=(-5.0, 5.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--ind", nargs=2, type=float, default=(-5.0, 5.0), metavar=("LOW", "HIGH"))
    ap.add_argument("--n", type=int, default=100, help="grid points per axis (default 100)")
    ap.add_argument("--rtc", action="store_true", help="RTC SPNE instead of SPNE")
    ap.add_argument("--csv", help="write the code grid to this CSV")
    ap.add_argument("--png", help="write the heatmap to this PNG (needs matplotlib)")
    a = ap.parse_args(argv)

    t0 = time.time()
    Inevec, Indvec, codes = compute_grid(a.payoffs, a.W, a.ine, a.ind, a.n, a.n, a.rtc)
    counts = {}
    for row in codes:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    print("computed %dx%d grid in %.1f s" % (len(Inevec), len(Indvec), time.time() - t0))
    for v in sorted(counts):
        print("  code %2d  %-34s %5d cells" % (v, describe_code(v), counts[v]))

    if a.csv:
        with open(a.csv, "w") as fh:
            fh.write("inequality,individualism,code,outcomes\n")
            for i, x in enumerate(Inevec):
                for j, y in enumerate(Indvec):
                    fh.write("%.17g,%.17g,%d,%s\n" % (x, y, codes[i][j], describe_code(codes[i][j])))
        print("wrote", a.csv)
    if a.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
        grid = [[codes[i][j] for i in range(len(Inevec))] for j in range(len(Indvec))]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(grid, origin="lower", cmap=ListedColormap(COLORS), vmin=-0.5, vmax=15.5,
                  extent=(Inevec[0], Inevec[-1], Indvec[0], Indvec[-1]), aspect="auto",
                  interpolation="nearest")
        ax.set_xlabel("Inequality")
        ax.set_ylabel("Individualism")
        ax.set_title("Supported subgame outcomes, W = %g (%s)" % (a.W, "RTC SPNE" if a.rtc else "SPNE"))
        ax.legend(handles=[Patch(color=COLORS[v], label="%d: %s" % (v, describe_code(v)))
                           for v in sorted(counts)],
                  loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9, frameon=False)
        fig.tight_layout()
        fig.savefig(a.png, dpi=150)
        print("wrote", a.png)


if __name__ == "__main__":
    _main()
