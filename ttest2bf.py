"""
Very simple Python module to calculate JZS Bayes Factor for a two-sample t-test given t and sample sizes.

See Rouder et al, 2009, Psychon Bull Rev for details.

The Bayesian two-sample T-Test
----------------------------
"""


from scipy.integrate import quad
from math import pi, exp

def ttest2bf(t, nx, ny, r=0.707):
  """
  This function calculates JZS Bayes Factor for a two-sample t-test given t
  and sample sizes.
  This quantifies the evidence in favour of the alternative hypothesis.

  Args:
      t (float): 
          Two-sample T-Test statistic value.
      nx (int):
          Size of sample x.
      ny (int):
          Size of sample y.
      r (float):
          Scale factor. Default 0.707.

  Returns:
      bf (float): 
          JZS Bayes Factor.
  """

  def F(g, nx, ny, r, t):
    return (1+(nx*ny/(nx+ny))*g*r**2)**(-1./2) * \
           (1 + t**2/((1+(nx*ny/(nx+ny))*g*r**2)*(nx+ny-2)))**(-(nx+ny-1)/2) *\
           (2*pi)**(-1./2) * g**(-3./2) * exp(-1/(2*g))

  I = quad(F, 0, float('Inf'), args=(nx, ny, r, t))[0]

  return I/(1 + t**2/(nx+ny-2))**(-(nx+ny-1)/2)
