from Environment import Evaluator
from configuration import configuration
# XXX this could be read from a JSON file instead?

if __name__ == '__main__':
    evaluation = Evaluator(configuration)
    evaluation.start()
    evaluation.plotResults(0)
