from torch.nn import Module

class Agent(Module):
    def act(self, *args, **kwargs):
        x = self.observation_to_tensor(*args, **kwargs)
        x = self(x)
        a = self.tensor_to_action(x)
        
        return a
    
    def observation_to_tensor(self):
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
    
    def tensor_to_action(self, x):
        raise NotImplementedError
    
    def train(self, batch):
        raise NotImplementedError
