

class Lover():
    weekends_with_you = 'My lover lover lover don\'t say no.\
                        I just wanna head home I don\'t feel so well.'

    def __init__(self, name, date_from='Someday'):
        self.name = name
        self.date_from = date_from
        self.memories = list()

    @classmethod
    def do_something(cls):
        print(cls.weekends_with_you)
        return cls.weekends_with_you

    def add_memory(self, memory):
        self.memories.append(memory)

    def recall_memory(self):
        print(self.memories)
        return self.memories


if __name__ == '__main__':
    # call classmethod without instance it.
    Lover.do_something()

    # call classmethod by object.
    someone_i_love = Lover('JC', '2020-10-20')
    someone_i_love.do_something()


