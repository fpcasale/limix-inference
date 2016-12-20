import os
import lim
import numpy as np

from numpy.testing import assert_almost_equal

from numpy_sugar.linalg import economic_qs_linear

from limix_inference.glmm import ExpFamEP
from limix_inference.lik import BinomialProdLik
from limix_inference.link import LogitLink

def test_binomial_delta():
    random = np.random.RandomState(0)
    G = random.randn(4, 5)
    K = G.dot(G.T)
    ntri = [5, 10, 3, 15]
    nsuc = [2, 1, 3, 5]
    M = np.ones((4, 1))

    # pheno = BinomialPhenotype(np.asarray(nsuc), np.asarray(ntri))
    # s = scan(pheno, G, K=K, covariates=covariates, progress=False)
    # print(s.pvalues())


    (Q, S0) = economic_qs_linear(G)

    lik = BinomialProdLik(np.asarray(ntri), LogitLink())
    lik.nsuccesses = np.asarray(nsuc)

    ep = ExpFamEP(lik, M, Q[0], Q[1], S0 + 1)

    ep.beta = np.array([1.])
    ep.v = 1.
    # ep.delta = 0
    # print(ep.lml())

    ep.delta = 1
    print(ep.lml())
    # assert_almost_equal(ep.lml(), -2.3202659215368935)

if __name__ == '__main__':
    __import__('pytest').main([__file__, '-s'])
