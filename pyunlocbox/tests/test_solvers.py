#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test suite for the functions module of the pyunlocbox package.
"""

import unittest
import numpy as np
import numpy.testing as nptest
from pyunlocbox import functions, solvers


class FunctionsTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_forward_backward(self):
        """
        Test forward-backward solver with L2-norm and dummy functions.
        """
        y = [4, 5, 6, 7]
        x0 = np.zeros(len(y))
        solver = solvers.forward_backward(method='FISTA', lambda_=1, gamma=1)
        param = {'x0': x0, 'solver': solver}
        param['absTol'] = 1e-5
        param['verbosity'] = 'none'

        # L2-norm prox and dummy gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.dummy()
        sol = [3.99996922, 4.99996153, 5.99995383, 6.99994614]
        ret = solvers.solve([f1, f2], **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ABS_TOL')
        self.assertEqual(ret['niter'], 10)

        # Dummy prox and L2-norm gradient.
        f1 = functions.dummy()
        f2 = functions.norm_l2(y=y)
        sol = y
        ret = solvers.solve([f1, f2], **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'ABS_TOL')
        self.assertEqual(ret['niter'], 1)

        # L2-norm prox and L2-norm gradient.
        f1 = functions.norm_l2(y=y)
        f2 = functions.norm_l2(y=y)
        sol = [3.99904855, 4.99881069, 5.99857282, 6.99833496]
        ret = solvers.solve([f1, f2], **param)
        nptest.assert_allclose(ret['sol'], sol)
        self.assertEqual(ret['crit'], 'MAX_IT')


suite = unittest.TestLoader().loadTestsFromTestCase(FunctionsTestCase)


def run():
    unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == '__main__':
    run()
