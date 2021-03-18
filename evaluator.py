from typing import AsyncIterable
from primitives import env as penv
from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from pyrsistent import pmap,plist
import torch
import numpy as np
import matplotlib.pyplot as plt
import time

#from primitives import PRIMITIVES


class Env(dict):
    "An environment: a dict of {'var': val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)


class Lambda(object):
    "A user-defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args): 
        print(self.body)
        return [ evaluate(self.body, Env(self.parms, args, self.env)) , {}]
        #return eval(self.body, {}, Env(self.parms, args, self.env))

def standard_env() -> Env:
    env = Env()
    env.update(pmap(penv))
    #print(env.update(penv)) #primitive env
    #env = env.update(pmap(penv)) #primitive env
    print("env is ", env)
    return env



    #env = pmap(penv) #primitive env
    #env = env.update({'alpha' : ''})



def evaluate(ast, envr=None): #TODO: add sigma, or something

    #print("ast is", ast)

    #print(len(ast))

    #print(type(ast))

    #print(ast[0])

    alpha0 = '0' #default address


    if envr is None:
        envr = standard_env()


    #print(envr)

    #### not needed?
    #PROCS = {}
    #for i in range(len(ast)-1):
    #    proc = ast[i]
    #    proc_name, proc_arg_names, proc_expr = proc[1], proc[2], proc[3]
    #    PROCS[proc_name] = (proc_arg_names,proc_expr)
    ####




    def eval(expr, sigma, envr):

        #print("the current expr",expr)

        #print("envr is: ", envr)
        
        if is_const(expr, envr):
            if type(expr) in [int, float, bool]:
                expr = torch.Tensor([expr]).squeeze()
            return expr, sigma

        elif is_var(expr, envr): #variable reference
            return envr.find(expr)[expr], sigma
            #return envr[expr], sigma
        
        elif not isinstance(expr,list):
            return expr, sigma

        op, *args = expr



        if is_fn(op,envr):
            #print("op is:", op)
            #print("args is:", args)
            (params, body) = args
            return Lambda(params,body,envr), sigma

        elif is_if(expr,envr):
            cond_expr, true_expr, false_expr = expr[1], expr[2], expr[3]
            cond_value, sigma = eval(cond_expr, sigma, envr)
            if cond_value:
                return eval(true_expr, sigma, envr)
            else:
                return eval(false_expr, sigma, envr)


        elif is_sample(expr,envr):
            dist_expr = expr[1]
            dist_obj, sigma = eval(dist_expr,sigma,envr)
            # return sample from distribution object
            return dist_obj.sample(), sigma


        elif is_observe(expr,envr):
            dist_expr, obs_expr = expr[1], expr[2]
            dist_obj, sigma = eval(dist_expr,sigma,envr)
            obs_value, sigma = eval(obs_expr,sigma,envr)
            # update trace total likelihood for importance sampling
            #sigma['log_W'] = sigma['log_W'] + dist_obj.log_prob(obs_value)
            return obs_value, sigma
            ## let no longer exists


       # elif is_let(expr, envr):
       #     var_name, sub_expr, final_expr = expr[1][0], expr[1][1], expr[2]
       #     var_value, sigma = eval(sub_expr, sigma, envr)
       #     return eval(final_expr, sigma, {**envr, var_name: var_value})
        
        else:
            proc=eval(op,sigma,envr)[0]

            print("proc: ", proc )
            print(type(proc))
            
            if is_var(proc, envr): #variable reference
                return envr.find(proc)[proc], sigma
            elif is_const(proc, envr):
                if type(proc) in [int, float, bool]:
                    proc = torch.Tensor([proc]).squeeze()
                return proc, sigma

            vals = [ eval(arg,sigma,envr) for arg in args]
            #print("vals: ", vals)
            return proc(*vals)

        #else:
        #    proc_name = expr[0]
        #    consts = []
        #    for i in range(1,len(expr)):
        #        const, sigma = eval(expr[i],sigma,envr)
        #        consts.append(const)
        #    if proc_name in PROCS:
        #        proc_arg_names, proc_expr = PROCS[proc_name]
        #        new_envr = {**envr}
        #        for i, name in enumerate(proc_arg_names):
        #            new_envr[name] = consts[i]
        #        return eval(proc_expr, sigma, new_envr)
        #    else:
        #        #return PRIMITIVES[proc_name](*consts), sigma
        #        return PRIMITIVES[proc_name](*consts), sigma

    #give a start adderess
    return eval([ast,alpha0],{'log_W': 0.},envr)            
    #return eval(ast[-1], {'log_W': 0.}, {})

def is_const(expr, envr):
    print("here:",expr)
    print(type(expr))
    #print("envr is: ",envr)
    #return type(expr) not in [tuple,list,dict] and expr not in PRIMITIVES and expr not in envr
    return type(expr) not in [tuple,list,dict] and expr not in envr

def is_var(expr, envr):
    return type(expr) not in [tuple,list,dict] and expr in envr
def is_let(expr, envr):
    return expr[0] == "let"
def is_if(expr, envr):
    return expr[0] == "if"
def is_sample(expr, envr):
    return expr[0] == "sample"
def is_observe(expr, envr):
    return expr[0] == "observe"

def is_fn(expr,envr):
    return expr == "fn"


def get_stream(exp):
    while True:
        yield evaluate(exp)


def run_deterministic_tests():
    
    
    for i in range(1,14):

        exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/tests/deterministic/test_{}.daphne'.format(i)])
        print("\n \n", exp, "\n \n")
        
        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)

        print("ret", ret)

        rett = ret[0]

        print("rett", rett)
        
        print("return value", rett('0'))

        print("truth", truth)

        '''
        try:
            assert(is_tol(ret, truth))
        except:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        '''

        print('FOPPL Tests passed')

    

    for i in range(1,13):

        exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/tests/hoppl-deterministic/test_{}.daphne'.format(i)])
        
        
        print("hoppl deterministic test {} \n \n".format(i), exp, "\n \n")

        
        truth = load_truth('programs/tests/hoppl-deterministic/test_{}.truth'.format(i))
        ret = evaluate(exp)

        print("return value", ret)
        print("truth", truth)

        
        #try:
        #    assert(is_tol(ret, truth))
        #except:
        #    raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,exp))
        
        print('Test passed')
        
        

    print('All deterministic tests passed')
    






def run_probabilistic_tests():
    
    num_samples=1e4
    max_p_value = 1e-2
    
    for i in range(1,7):
        exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(exp)
        
        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')    



if __name__ == '__main__':
    
    run_deterministic_tests()
    #run_probabilistic_tests()


    '''

    N = 1000
    


    # program 2

    exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/1.daphne'])
    print("\n running Program 2")
    start = time.time()
    vals = [ evaluate(exp) for i in range(N) ]
    end = time.time()

    print("\n Program 2 mean: ", np.mean(samples))
    print("\n Program 2 variance: ", np.var(samples))
    print("\n Run time is: ", start-end)

    plt.hist(vals)
    plt.savefig('../HW5/tex/program2_hist.png')




    # program 3

    exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/2.daphne'])
    print("\n running Program 3")
    start = time.time()
    vals = [ evaluate(exp) for i in range(N) ]
    end = time.time()

    print("\n Program 3 mean: ", np.mean(samples))
    print("\n Program 3 variance: ", np.var(samples))
    print("\n Run time is: ", start-end)

    plt.hist(vals)
    plt.savefig('../HW5/tex/program3_hist.png')



    # program 4

    exp = daphne(['desugar-hoppl', '-i', '../HW5/programs/3.daphne'])
    print("\n running Program 4")
    start = time.time()
    vals = [ evaluate(exp) for i in range(N) ]
    end = time.time()
    print("\n Program 4 mean: ", np.mean(samples))
    print("\n Program 4 variance: ", np.var(samples))
    print("\n Run time is: ", start-end)
    
    plt.hist(vals)
    plt.savefig('../HW5/tex/program4_hist.png')

    '''

    #for i in range(1,4):
    #    print(i)
    #    exp = daphne(['desugar-hoppl', '-i', '../../HW5/programs/{}.daphne'.format(i)])
    #    print('\n\n\nSample of prior of program {}:'.format(i))
    #    print(evaluate(exp))        
