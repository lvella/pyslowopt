import math

class ConvergenceTester:
    def __init__(self, F, abs_tolerance, rel_tolerance):
        self.F = F
        self.abs_tolerance = abs_tolerance
        self.rel_tolerance = rel_tolerance
        self.first_test_passed = False

    def test_convergence(self, oldX, newX):
        new_val = self.F(*newX)
        abs_dif = math.fabs(self.F(*oldX) - new_val)

        abs_conv = abs_dif < self.abs_tolerance
        rel_conv = (abs_dif / max(math.fabs(new_val), 1e-10)) < self.rel_tolerance

        return abs_conv and rel_conv

    def test_constaints(self, h_list, g_list, X):
        for h in h_list:
            hval = math.fabs(h(*X))
            if hval > self.rel_tolerance:
                return False
        for g in g_list:
            gval = g(*X)
            if gval > self.rel_tolerance:
                return False
        return True

    def has_converged(self, *args):
        test = self.test_convergence(*args)
        if test and self.first_test_passed:
            return True

        self.first_test_passed = test
        return False
