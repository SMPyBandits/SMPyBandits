from Environment import Evaluator
# XXX could this be read from a JSON file instead?
from configuration import configuration

if __name__ == '__main__':
    evaluation = Evaluator(configuration)
    evaluation.start()
    evaluation.plotResults(0)
