"""
author: Julien Seznec
Rested rotting arm, i.e. arms with mean value which decay at each pull
"""
try:
    from .Arm import Arm
    from .Gaussian import UnboundedGaussian as Gaussian
except ImportError:
    from Arm import Arm
    from Gaussian import UnboundedGaussian as Gaussian


class RestedRottingArm(Arm):
  def __init__(self, decayingFunction, staticArm):
    self.decayingFunction = decayingFunction
    # It provides the mean of the arm after n pulls. EXCEPT for truncated distributions where it is the mean of the untrucated distributions
    self.arm = staticArm
    self.pull_count = 0
    self.arm.set_mean_param(self.decayingFunction(self.pull_count))
    self.mean = self.arm.mean


  def draw(self, t = None):
    draw = self.arm.draw(t)
    current_mean = self.mean
    self.pull_count += 1
    self.arm.set_mean_param(self.decayingFunction(self.pull_count))
    self.mean = self.arm.mean
    assert current_mean >= self.mean, "Arm has increased."
    return draw


# DECAYING FUNCTIONS
def constant(x, mu):
  return mu

def abruptSingleDecay(x,mu, switchPoint):
  return mu if x < switchPoint else -mu

if __name__ == '__main__':
  gaussian = Gaussian(0,1)
  rotting_gaussian = RestedRottingArm(lambda n: 0 if n>10 else 1, gaussian)
  for i in range(100):
    print(rotting_gaussian.draw())

