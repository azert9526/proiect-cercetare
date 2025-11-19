import random
import string
import time
from hll import cardinality_estimation

def generate_random_strings(n=10000, length=30):
    alphabet = string.ascii_letters + string.digits
    return [
        ''.join(random.choices(alphabet, k=length))
        for _ in range(n)
    ]


random.seed(time.time())
print(cardinality_estimation(generate_random_strings(n=80000), p=12))
