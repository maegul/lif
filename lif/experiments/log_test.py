


# +
import time
import multiprocessing as mp
from multiprocessing.pool import AsyncResult

from typing import Optional
from pathlib import Path
# -
# +
def test_func(n, a: int, b: int, stdout: Optional[Path]=None):

	time.sleep(2)
	f = open(stdout) if stdout is not None else None
	for i in range(a, b):
		print(f'{n}: {i}', file=f)
# -
# +
n_procs=3
# -
# +
pool = mp.Pool(processes=n_procs)

for i in range(7):

	time.sleep(0.5)
	pool.apply_async(
		func=test_func,
		kwds={
			'n': i,
			'a': i,
			'b': 10*i,
			# 'stdout': Path(f'./tl{(i%3)+1}')
		}
		)

	time.sleep(0.5)

pool.close()
pool.join()
# -
