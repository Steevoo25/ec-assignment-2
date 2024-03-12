import argparse
import sexpdata as sex
import math
import numpy as np
#from genetic_programmng import ga

SAMPLE_EXP_1 = "(mul (add 1 2) (log 8))"
SAMPLE_EXP_2 = "(max (data 0) (data 1))"

# adds 2 expressions
def add(exp1, exp2):
    return float(evaluate(exp1)) + float(evaluate(exp2))

# Subtracts exp2 from exp1
def sub (exp1, exp2):
    return float(evaluate(exp1)) - float(evaluate(exp2))
    
# Multiplies 2 expressions
def mul(exp1, exp2):
    return float(evaluate(exp1)) * float(evaluate(exp2))

# Divides exp1 by exp 2 as long as exp2 is not 0
def div(exp1, exp2):
    exp2_eval = float(evaluate(exp2))
    exp1_eval = float(evaluate(exp2))
    if exp2_eval == 0:
        return 0
    else:
        return exp1_eval / exp2_eval
        
# Returns exp1 to the power of exp2
def pow(exp1, exp2):
    return float(evaluate(exp1)) ** float(evaluate(exp2))

# Returns the square root of exp1
def sqrt(exp1):
    exp1_eval = float(evaluate(exp1))
    if exp1_eval < 0:
        return 0
    return math.sqrt(exp1_eval)

# Takes log2 of 1 expression
def log(exp1):
    exp1_val = float(evaluate(exp1))
    if exp1_val > 0:
        return math.log2(exp1_val)
    else:
        return 0

# returns e^ exp1
def exp(exp1):
    exp1_val = float(evaluate(exp1))

    return math.e ** exp1_val

#returns larger value of exp1 and exp2
def max(exp1, exp2) -> float:
    exp1_eval = float(evaluate(exp1))
    exp2_eval = float(evaluate(exp2))
    
    if exp1_eval > exp2_eval :
        return exp1_eval
    else:
        return exp2_eval

# Returns exp3 if exp1 <= exp2, otherwise returns exp4
def ifleq(exp1, exp2, exp3, exp4):
    if float(evaluate(exp1)) <= float(evaluate(exp2)):
        return evaluate(exp3)
    else:
        return evaluate(exp4)

# Returns the element at exp1th element (floored, absed and modded)
# n is size of input vector x
def data(exp1) -> int:
    eval = float(evaluate(exp1))
    floor = math.floor(eval)
    index = np.abs(floor)
    if not index == 0:
        index = index % n 
    print(index, x[index])
    return x[index]

# Returns the difference between 2 elements
def diff(exp1, exp2) -> int:
    exp1_data = data(exp1)
    exp2_data = data(exp2)
    return exp1_data - exp2_data

# Returns the average of a range between 2 indecies
def avg(exp1, exp2):

    k = np.abs(math.floor(evaluate(exp1))) % n
    l = np.abs(math.floor(evaluate(exp2))) % n
    print("k , l" , k , l)
    if k == l:
        return 0
    
    difference = np.abs(k - l) + 1
    factor = 1/difference
    sum = 0
    
    if k < l: #exp1 to exp2
        for i in range(k, l+1):
            sum += data(i)
    else: # exp2 to exp1
        for i in range(l, k):
            sum += data(i)
            
    return float(factor * sum)

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
        #print("operator", operator,"operands",  operands)
        try:
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
                return data(*operands)
            elif operator == 'diff':
                return diff(*operands)
            elif operator == 'avg':
                return avg(*operands)
        except OverflowError:
            print("Value too large for operation: ", operator, operands, " returning 0 for this step")
            return 0

def open_training_data(training_data):
    x_values = []
    y_values = []
    # open file
    with open(training_data, 'r') as training_data_file:
        for line in training_data_file: 
            # split by tab
            values = line.split('\t')
            x_values.append(values[:-1])
            y_values.append(values[-1])
            
    return x_values, y_values

def squared_error(sexp):
    difference = y - evaluate(sexp)
    return difference ** 2

def calculate_fitness(e, n, m, training_data):
    x_values, y_values = open_training_data(training_data)
    total = 0
    factor = 1/m
    for i in range(0, m-1): # as x is global, update here
        x = x_values[i]
        y = y_values[i]
        total += squared_error(e)
            
        print(factor * total)

def get_args(args) -> list:
    # Access the arguments
    # q1+
    question = getattr(args, 'question', None)
    expr = getattr(args, 'expr', None)
    n = getattr(args, 'n', None)
    x = getattr(args, 'x', None)
    # q2+
    m = getattr(args, 'm', None)
    data = getattr(args, 'data', None)
    # q3+
    pop_size = getattr(args, 'lambda', None)
    time_budget = getattr(args, 'time_budget', None)
    return question, expr, n, x, m, data, pop_size, time_budget


def select_question(args):
    
    # Extract arguments
    q, e, n, x, m, training_data, pop_size, time_budget  = get_args(args)

    if q == 1:
        # cast arguments
        e = sex.loads(e)
        n = int(n)
        x = [float(num) for num in x.split(' ')]
        result = evaluate(e)
        print(result)
    if q == 2:
        # cast arguments
        e = sex.loads(e)
        n = int(n)
        m = int(m)
        result = calculate_fitness(e, n, m, training_data)
        
    if q == 3:
        pop_size = int(pop_size)
        time_budget = float(time_budget)
        result = ga()
    


def main():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Example script to demonstrate argparse')

    # Add arguments
    parser.add_argument('-question', help='question number')
    parser.add_argument('-expr', help='an expression')
    parser.add_argument('-n', help='dimension of unput vector n')
    parser.add_argument('-x', help='input vector')
    
    parser.add_argument('-m', help='Size of training data')
    parser.add_argument('-data', help='filename containing training data')
    
    parser.add_argument('-lambda', help='Population Size')
    parser.add_argument('-time_budget', help='number of seconds to run algorithm')
    
    # Parse the command line arguments
    args = parser.parse_args()
    select_question(args)
    
    return 

if __name__ == "__main__":
    main()