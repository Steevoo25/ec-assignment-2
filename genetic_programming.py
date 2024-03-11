import argparse
import sexpdata as sex
import math
import numpy as np

SAMPLE_EXP_1 = "(mul (add 1 2) (log 8))"
SAMPLE_EXP_2 = "(max (data 0) (data 1))"

# adds 2 expressions
def add(exp1, exp2):
    return evaluate(exp1) + evaluate(exp2)

# Subtracts exp2 from exp1
def sub (exp1, exp2):
    return evaluate(exp1) - evaluate(exp2)
    
# Multiplies 2 expressions
def mul(exp1, exp2):
    return evaluate(exp1) * evaluate(exp2)

# Divides exp1 by exp 2 as long as exp2 is not 0
def div(exp1, exp2):
    exp2_eval = evaluate(exp2)
    if exp2_eval == 0:
        return 0
    else:
        return evaluate(exp1) / exp2_eval
        
# Returns exp1 to the power of exp2
def pow(exp1, exp2):
    return evaluate(exp1) ** evaluate(exp2)
# CHECK FOR NEGATIVE?
def sqrt(exp1):
    return math.sqrt(evaluate(exp1))

# Takes log2 of 1 expression
def log(exp1):
    # ADD UNDEFINED CASE
    return math.log2(evaluate(exp1))
# returns e^ exp1
def exp(exp1):
    return math.e ** evaluate(exp1)

#returns larger value of exp1 and exp2
def max(exp1, exp2):
    return np.max((evaluate(exp1), evaluate(exp2)))

def ifleq(exp1, exp2, exp3, exp4):
    if evaluate(exp1) <= evaluate(exp2):
        return evaluate(exp3)
    else:
        return evaluate(exp4)

def data(exp1, n):
    return np.abs(math.floor(evaluate(exp1))) % n

def diff(exp1, exp2):
    return evaluate(data(exp1)) -  evaluate(data(exp2))

def avg(exp1, exp2):
    return evaluate(exp1), evaluate(exp2)

def evaluate(sexp):
    # if its an atom
    if isinstance(sexp, int):
        return sexp
    elif isinstance(sexp, sex.Symbol):
        # Handle symbols?
        print("its a symbol", sexp)
        return None
    elif isinstance(sexp, list):
    
        operator = str(sex.car(sexp))
        operands = sex.cdr(sexp)
        
        if operator == 'add':
            return add(*operands)
        elif operator == 'sub':
            return sub(*operands)
        elif operator == 'mul':
            return mul(*operands)
        elif operator == 'div':
            return div(*operands)
        elif operator == 'pow':
            return pow(*operands)
        elif operator == 'sqrt':
            return sqrt(*operands)
        elif operator == 'log':
            return log(*operands)
        elif operator == 'exp':
            return exp(*operands)
        elif operator == 'max':
            return max(*operands)           
        elif operator == 'ifleq':
            return ifleq(*operands)
        elif operator == 'data':
            return x[data(*operands, n)]
        elif operator == 'diff':
            return diff(*operands)
        elif operator == 'avg':
            return avg(*operands)

def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse')

    # Add arguments
    parser.add_argument('-question', help='question number')
    parser.add_argument('-expr', help='an expression')
    parser.add_argument('-n', help='dimension of unput vector n')
    parser.add_argument('-x', help='input vector')
    
    # Parse the command line arguments
    args = parser.parse_args()

    # Access the arguments
    question = args.question
    expr = args.expr
    n = args.n
    x = args.x
    
    return question, expr, n , x

if __name__ == "__main__":

    q, e, n, x = main()
    print(q, e, n, x)
    
    if q == 1:
        evaluate(e)