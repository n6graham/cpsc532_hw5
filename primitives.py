import torch
import torch.distributions as dist
from copy import deepcopy
from collections.abc import Iterable




class Normal(dist.Normal):
    
    def __init__(self, alpha, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc, torch.nn.functional.softplus(self.optim_scale))
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
        

def push_addr(alpha, value):
    return alpha + value



def add(alpha,a,b):
    return torch.add(a,b)
def subtract(alpha,a,b):
    return torch.subtract(a,b)
def multiply(alpha,a,b):
    return torch.multiply(a,b)
def divide(alpha,a,b):
    return torch.divide(alpha,a,b)
def gt(alpha,a,b):
    return torch.Tensor([a>b])
def lt(alpha,a,b):
    return torch.Tensor([a<b])
def eq(alpha,a,b):
    return torch.Tensor([a==b])
def compare_and(alpha,a,b):
    return torch.Tensor([a and b])
def compare_or(alpha,a,b):
    return torch.Tensor([a or b])
def sqrt(alpha,a):
    return torch.sqrt(a)
def tanh(alpha,a):
    return torch.tanh(a)
def first(alpha,data):
    return data[0]
def second(alpha,data):
    return data[1]
def rest(alpha,data):
    return data[1:]
def last(alpha,data):
    return data[-1]
def nth(alpha,data, index):
    return data[index]
def conj(alpha,data, el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([data, el], dim=0)
def cons(alpha,data, el):
    if len(el.shape) == 0: el = el.reshape(1)
    return torch.cat([el, data], dim=0)


#not sure how I should add address here, figure out later    
def vector(*args):
    # sniff test: if what is inside isn't int,float,or tensor return normal list
    if type(args[0]) not in [int, float, torch.Tensor]:
        return [arg for arg in args]
    # if tensor dimensions are same, return stacked tensor
    if type(args[0]) is torch.Tensor:
        sizes = list(filter(lambda arg: arg.shape == args[0].shape, args))
        if len(sizes) == len(args):
            return torch.stack(args)
        else:
            return [arg for arg in args]
    raise Exception(f'Type of args {args} could not be recognized.')
def hashmap(*args):
    result, i = {}, 0
    while i<len(args):
        key, value  = args[i], args[i+1]
        if type(key) is torch.Tensor:
            key = key.item()
        result[key] = value
        i += 2
    return result


def get(addr,struct, index):
    if type(index) is torch.Tensor:
        index = index.item()
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    return struct[index]
def put(addr,struct, index, value):
    if type(index) is torch.Tensor:
        index = int(index.item())
    if type(struct) in [torch.Tensor, list, tuple]:
        index = int(index)
    result = deepcopy(struct)
    result[index] = value
    return result
def bernoulli(addr,p, obs=None):
    return torch.distributions.Bernoulli(p)
def beta(addr,alpha, beta, obs=None):
    return torch.distributions.Beta(alpha, beta)
def normal(addr,mu, sigma):
    return torch.distributions.Normal(mu, sigma)
def uniform(addr,a, b):
    return torch.distributions.Uniform(a, b)
def exponential(addr,lamb):
    return torch.distributions.Exponential(lamb)
def discrete(addr,vector):
    return torch.distributions.Categorical(vector)
def gamma(addr,concentration, rate):
    return torch.distributions.gamma.Gamma(concentration, rate)
def dirichlet(addr,concentration):
    return torch.distributions.dirichlet.Dirichlet(concentration)
def dirac(alpha,point):
    class Dirac:
        def __init__(self, point):
            self.point = point
        def sample(self):
            return self.point
        def log_prob(self, obs):
            return torch.distributions.normal.Normal(self.point,1e-3).log_prob(obs)
            value = 0. if obs == self.point else -float('inf')
            return torch.Tensor([value]).squeeze()
    return Dirac(point)
def transpose(addr,tensor):
    return tensor.T
def repmat(addr,tensor, size1, size2):
    if type(size1) is torch.Tensor: size1 = int(size1.item())
    if type(size2) is torch.Tensor: size2 = int(size2.item())
    return tensor.repeat(size1, size2)
def matmul(addr,t1, t2):
    return t1.matmul(t2)



env = {
        "normal" : Normal,
        "push-address" : push_addr,
        "+": add,
        "-": subtract,
        "*": multiply,
        "/": divide,
        ">": gt,
        "<": lt,
        "=": eq,
        "sqrt": sqrt,
        "first": first,
        "second": second,
        "rest": rest,
        "last": last,
        "nth": nth,
        "append": conj,
        "and": compare_and,
        "or": compare_or,
        "conj": conj,
        "cons": cons,
        "vector": vector,
        "hash-map": hashmap,
        "list": list,
        "get": get,
        "put": put,
        "flip": bernoulli,
        "beta": beta,
        "normal": normal,
        "uniform": uniform,
        "exponential": exponential,
        "discrete": discrete,
        "gamma": gamma,
        "dirichlet": dirichlet,
        "dirac": dirac,
        "mat-transpose": transpose,
        "mat-add": add,
        "mat-tanh": tanh,
        "mat-repmat": repmat,
        "mat-mul": matmul,
        #"if": lambda cond, v1, v2: v1 if cond else v2 # for graph based sampling
    }








