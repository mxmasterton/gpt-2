import torch
import tiktoken

# -------------------------------------------------------------

class DataLoader:
    
    def __init__(self, B, T):
        self.B = B
        self.T = T
        
        with open("data/shakespeare.txt", "r") as f:
            text = f.read()
            
        enc = tiktoken.get_encoding("gpt2")
        tokens = self.encode(text)
        
        self.tokens = torch.encode(tokens)
        
        print("loaded {} tokens".format(len(tokens)))
        print("1 epoch = {} batches".format(len(self.tokens) // (B*T)))
            