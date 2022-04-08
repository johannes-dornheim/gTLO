from gym import Wrapper
import numpy as np



class DeepDrawingMORLWrapper(Wrapper):
    # MORL environment
    def __init__(self, env, r_terms, prohibit_negative_rewards=True):
        super().__init__(env)
        for r_term in r_terms:
            assert r_term in ['rt_feeding', 'rt_thickness', 'rt_v_mises']
        self.r_terms = r_terms
        self.prohibit_negatives = prohibit_negative_rewards

    def step(self, action):
        o, r, done, info = self.env.step(action)
        r = np.array([info[r_term] for r_term in self.r_terms])
        if self.prohibit_negatives and np.any(r < 0):
            r = np.zeros(len(self.r_terms))
        return o, r, done, info

