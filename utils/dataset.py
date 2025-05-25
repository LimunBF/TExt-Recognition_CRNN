class LabelEncoder:
    def __init__(self, alphabet):
        # blank token = 0, then 1..len(alphabet)
        self.char2idx = {char: idx + 1 for idx, char in enumerate(alphabet)}
        self.idx2char = {idx + 1: char for idx, char in enumerate(alphabet)}
        self.blank = 0

    def encode(self, text):
        return [self.char2idx[char] for char in text]

    def decode(self, indices):
        chars = []
        for i, idx in enumerate(indices):
            # skip blanks & repeated tokens
            if idx != self.blank and (i == 0 or idx != indices[i - 1]):
                chars.append(self.idx2char.get(idx, ""))
        return ''.join(chars)
    
