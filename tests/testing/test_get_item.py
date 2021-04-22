
class Property:
    def __init__(self, a, b):
        self.value = {a:b}
    def __getitem__(self, index):
        return self.value[index]

p = Property('a', 2)
print(p['a'])