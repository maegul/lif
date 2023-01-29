---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
from scipy import stats as st
from scipy import optimize as opt
```

```python
import ipywidgets as ipw
```

```python
import matplotlib.pyplot as plt
```

```python
# Distribution requirements
# From table 2 Saul and Humphrey 1990
mu=33
std=5 * (18**0.5)
min_val, max_val = 10, 75
```

```python
print(std)
```

```python
def saul_humph_max_f1(alpha, beta, loc, scale, dist_type='skewnorm'):
    if dist_type=='skewnorm':
        dist=st.skewnorm(alpha, loc=loc, scale=scale)
    elif dist_type=='skewcauchy':
        dist=st.skewcauchy(alpha, loc=loc, scale=scale)
    elif dist_type=='beta':
        dist=st.beta(alpha=alpha, beta=beta, loc=loc, scale=scale)
    else:
        raise ValueError('Invalid dist_type')
    s=dist.rvs(5000)
    
    return s, dist
```

```python
def saul_humph_dist_view(alpha, beta, loc, scale, bins=50, dist_type='skewnorm'):
    s,dist = saul_humph_max_f1(alpha, beta, loc, scale)
    s_samp = dist.rvs((60,100))
    samp_mu, samp_se = s_samp.mean(), st.sem(s_samp).mean()
    plt.hist(s, bins=bins)
    plt.title(f'µ={dist.mean():.1f}, std={dist.std():.1f}, min/max={s.min():.1f}/{s.max():.1f}, mu/se={samp_mu:.1f}/{samp_se:.1f}')
    for v in (11, 107):
        plt.axvline(v, 0, 1, c='k', linestyle=':')
```

```python
ipw.interact(saul_humph_dist_view, 
             alpha=(0, 20),
             beta=(0, 20),
             loc=(0, 50, 1), 
             scale=(1, 50, 1),
             dist_type=[
                 'skewnorm',
                 'skewcauchy',
                 'beta'
             ]
            )
```

<!-- #region -->
**Good potential values**

*skewnorm*
*Just use the SE from the spot data in Saul_Humph, and scale down (linearly?) to grating range (3(63) to ~2)*

* 20 alpha
* 13 loc
* 26 scale


*Using the spot data ...* **probably the best**

*skewnorm*

* 10 alpha
* 19 loc
* 30 scale

**Or ... actually best ... more intuitively constrained**

* 10 alpha
* 24 loc
* 28 scale

*Beta*

* 20 alpha
* 0 beta
* 11 loc
* 28 scale
<!-- #endregion -->

```python
plt.figure(figsize=(12, 5))
plt.vlines(st.skewnorm(10, loc=25, scale=28).rvs(20), 0, 1)
plt.xlim(10, 100)
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```
