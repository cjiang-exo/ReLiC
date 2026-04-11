from numpy import array, clip
from pymultinest.run import run as run_pm
from scipy.stats import norm, truncnorm

class Priors():
	def __init__(self, prior_list):
		self.prior_list = prior_list
		self.prior_functions = []
		prior_mapping = {
            "UP": self._uniform,
            "NP": self._normal,
            "TNP": self._trunc_normal
			}
		for item in prior_list:
			prior_type = item[0]
			if prior_type in prior_mapping:
				f = lambda x, item=item, func=prior_mapping[prior_type]: func(x, *item[1:])
			else:
				raise ValueError(
				'Set prior names as "UP", "NP", or "TNP".'
			)
			self.prior_functions.append(f) 

	def _uniform(self, x, a, b):
		""" U(0, 1) --> U(a, b) """ 
		return a + x * (b - a) 
	
	def _normal(self, x, mean, std):
		""" U(0, 1) -->  N(mean, std) """
		x = clip(x, 1e-9, 1 - 1e-9) # within 6 sigmas
		return norm.ppf(x, loc=mean, scale=std)
    
	def _trunc_normal(self, x, a, b, mean, std):
		"""
		U(0, 1) --> U(a, b)*N(mean, std)
		"""
		a_new = (a - mean)/std
		b_new = (b - mean)/std
		x = clip(x, 1e-9, 1 - 1e-9) # within 6 sigmas
		return truncnorm.ppf(x, a_new, b_new, mean, std)   
			
	def get_priors(self, cube):
		return array([f(x) for f, x in zip(self.prior_functions, cube)])

def run_multinest(LogLikelihood, Prior, n_dims, **kwargs):
	kwargs['n_dims'] = n_dims  
	def SafePrior(cube, ndim, nparams):
		try:
			n_params = kwargs.get('n_params', n_dims)
			a = array([cube[i] for i in range(n_params)])
			b = Prior(a)
			for i in range(n_params):
				cube[i] = b[i]
		except Exception as e:
			import sys
			sys.stderr.write('ERROR in prior: %s\n' % e)
			sys.exit(1)
	
	def SafeLoglikelihood(cube, ndim, nparams, lnew):
		try:
			a = array([cube[i] for i in range(n_dims)])
			l = float(LogLikelihood(a)) 
			return l
		except Exception as e:
			import sys
			sys.stderr.write('ERROR in loglikelihood: %s\n' % e)
			sys.exit(1)
	
	kwargs['LogLikelihood'] = SafeLoglikelihood
	kwargs['Prior'] = SafePrior
	run_pm(**kwargs)
	
