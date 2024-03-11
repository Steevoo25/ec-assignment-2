import sexpdata as sex
import math

SAMPLE_EXP_1 = "(mul (add 1 2) (log 8))"
SAMPLE_EXP_2 = "(max (data 0) (data 1))"

def evaluate_exp(exp):
    print("Evaluating", exp)
# adds 2 expressions
def add(exp1, exp2):
    return exp1 + exp2
#
def log(exp1):
    # ADD UNDEFINED CASE
    return math.log2(exp1)

expression_1 = sex.loads(SAMPLE_EXP_1)
expression_2 = sex.loads(SAMPLE_EXP_2)

# get Operator = car()
# get Operand(s) = cdr()

def splitExpression(exp):
    
    try:
        print(sex.car(exp))
        evaluate_exp(exp)
        splitExpression(sex.cdr(exp))
    except IndexError: # Atoms have been reached
        print("End of expression reached")
    return 

splitExpression(expression_1)
#splitExpression(expression_2)