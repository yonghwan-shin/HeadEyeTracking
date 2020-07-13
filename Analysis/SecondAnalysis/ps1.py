from tqdm import tqdm
from time import sleep

from tqdm import trange

from tqdm.auto import trange
from time import sleep

t = tqdm(total=100)
for i in range(100):
    t.update(1)
    sleep(0.1)
t.close()