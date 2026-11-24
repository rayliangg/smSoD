from dataclasses import dataclass


@dataclass
class Config:
    m: int = 5
    k: int = 2
    lower: float = -200.0
    upper: float = 200.0

    pop_size: int = 200
    tour_size: int = 8
    cross_prob: float = 0.975
    max_gen: int = 2000

    split_rate: float = 0.5
    decrease_rate: float = 0.998
    min_bound: float = 1e-5

    seed: int = None
    gpu_min: int = 1 << 17     # use the gpu only when a step has at least this many elements
    group_crossover: bool = False

    @property
    def length(self):
        return self.m * self.k

    def check(self):
        if self.m < 1:
            raise ValueError("m must be >= 1")
        if self.k < 1 or self.k > 62:
            raise ValueError("k must be in [1, 62]")
        if self.pop_size < 2:
            raise ValueError("pop_size must be >= 2")
        if self.tour_size < 1 or self.tour_size > self.pop_size:
            raise ValueError("tour_size must be in [1, pop_size]")
        if self.gpu_min < 0:
            raise ValueError("gpu_min must be >= 0")
        if self.lower >= self.upper:
            raise ValueError("lower must be < upper")
