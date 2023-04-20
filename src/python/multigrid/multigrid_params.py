

# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class MultigridParams:
    def __init__(self, factor: int, cycles_num: int =4, pre_relaxation_iters: int = 2, post_relaxation_iters: int = 2):
        self.factor = factor
        self.cycles_num = cycles_num
        self.pre_relaxation_iters = pre_relaxation_iters
        self.post_relaxation_iters = post_relaxation_iters