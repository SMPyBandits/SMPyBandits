"""
author: Julien Seznec
Restless arm, i.e. arms with mean value which change at each round
"""
try:
    from . import Arm, Bernoulli, Binomial, UnboundedExponential, UnboundedGaussian, Constant, UnboundedPoisson
except ImportError:
    from Arm import Arm
    from Bernoulli import Bernoulli
    from Binomial import Binomial
    from Exponential import UnboundedExponential
    from Gaussian import UnboundedGaussian
    from Constant import Constant
    from Poisson import UnboundedPoisson

from math import sin


class RestlessArm(Arm):
    def __init__(self, rewardFunction, staticArm):
        self.reward = rewardFunction
        # It provides the mean of the arm after n pulls. EXCEPT for truncated distributions where it is the mean of the untrucated distributions
        self.arm = staticArm
        self.mean = self.arm.mean

    def draw(self, t):
        self.arm.set_mean_param(self.reward(t))
        self.mean = self.arm.mean
        draw = self.arm.draw(t)
        return draw


class RestlessBernoulli(RestlessArm):
    def __init__(self, rewardFunction):
        arm = Bernoulli(0)
        super(RestlessBernoulli, self).__init__(rewardFunction, arm)


class RestlessBinomial(RestlessArm):
    def __init__(self, rewardFunction, draws=1):
        arm = Binomial(0, draws)
        super(RestlessBinomial, self).__init__(rewardFunction, arm)


class RestlessConstant(RestlessArm):
    def __init__(self, rewardFunction):
        arm = Constant(0)
        super(RestlessConstant, self).__init__(rewardFunction, arm)


class RestlessExponential(RestlessArm):
    def __init__(self, rewardFunction):
        arm = UnboundedExponential(1)
        super(RestlessExponential, self).__init__(rewardFunction, arm)


class RestlessGaussian(RestlessArm):
    def __init__(self, rewardFunction, sigma=1):
        arm = UnboundedGaussian(0, sigma)
        super(RestlessGaussian, self).__init__(rewardFunction, arm)


class RestlessPoisson(RestlessArm):
    def __init__(self, rewardFunction, sigma=1):
        arm = UnboundedPoisson(0)
        super(RestlessPoisson, self).__init__(rewardFunction, arm)



if __name__ == '__main__':
    restless_bernoulli = RestlessBernoulli(lambda x :sin(x)**2)
    restless_gaussian = RestlessGaussian(lambda x :sin(x)**2)
    restless_binomial = RestlessBinomial(lambda x :sin(x)**2, draws=10)
    print([sin(t)**2 for t in range(50)])
    print([restless_gaussian.draw(t) for t in range(50)])
    print([restless_bernoulli.draw(t) for t in range(50)])
    print([restless_binomial.draw(t) for t in range(50)])
