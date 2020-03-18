"""
author: Julien Seznec
Rested rotting arm, i.e. arms with mean value which decay at each pull
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

class RestedRottingArm(Arm):
    def __init__(self, decayingFunction, staticArm):
        self.decayingFunction = decayingFunction
        # It provides the mean of the arm after n pulls. EXCEPT for truncated distributions where it is the mean of the untrucated distributions
        self.arm = staticArm
        self.pull_count = 0
        self.arm.set_mean_param(self.decayingFunction(self.pull_count))
        self.mean = self.arm.mean

    def draw(self, t=None):
        self.arm.set_mean_param(self.decayingFunction(self.pull_count))
        current_mean = self.mean
        self.mean = self.arm.mean
        draw = self.arm.draw(t)
        self.pull_count += 1
        self.arm.set_mean_param(self.decayingFunction(self.pull_count))
        self.mean = self.arm.mean
        assert current_mean >= self.mean, "Arm has increased."
        return draw


class RestedRottingBernoulli(RestedRottingArm):
    def __init__(self, decayingFunction):
        arm = Bernoulli(0)
        super(RestedRottingBernoulli, self).__init__(decayingFunction, arm)


class RestedRottingBinomial(RestedRottingArm):
    def __init__(self, decayingFunction, draws=1):
        arm = Binomial(0, draws)
        super(RestedRottingBinomial, self).__init__(decayingFunction, arm)


class RestedRottingConstant(RestedRottingArm):
    def __init__(self, decayingFunction):
        arm = Constant(0)
        super(RestedRottingConstant, self).__init__(decayingFunction, arm)


class RestedRottingExponential(RestedRottingArm):
    def __init__(self, decayingFunction):
        arm = UnboundedExponential(1)
        super(RestedRottingExponential, self).__init__(decayingFunction, arm)


class RestedRottingGaussian(RestedRottingArm):
    def __init__(self, decayingFunction, sigma=1):
        arm = UnboundedGaussian(0, sigma)
        super(RestedRottingGaussian, self).__init__(decayingFunction, arm)


class RestedRottingPoisson(RestedRottingArm):
    def __init__(self, decayingFunction, sigma=1):
        arm = UnboundedPoisson(0)
        super(RestedRottingPoisson, self).__init__(decayingFunction, arm)



if __name__ == '__main__':
    rotting_bernoulli = RestedRottingBernoulli(lambda n: 0 if n > 10 else 1)
    rotting_gaussian = RestedRottingGaussian(lambda n: 0 if n > 10 else 1)
    print([rotting_gaussian.draw() for _ in range(50)])
    print([rotting_bernoulli.draw() for _ in range(50)])