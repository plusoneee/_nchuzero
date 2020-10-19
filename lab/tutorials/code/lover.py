
# Import a module
# from <File Name> import <Class Name> or <Function Name>
from soulmate import SoulMate

class Lover(SoulMate):
    weekends_with_you = 'My lover lover lover don\'t say no.\
                         I just wanna head home I don\'t feel so well.'

    def __init__(self, name, in_common, date_from):
        super(Lover, self).__init__(name, in_common)
        self.date_from = date_from
        self.memories = list()

    def say_hi(self):
        print('%s say good morning.' % self.name)

    @classmethod
    def do_something(cls):
        print(cls.weekends_with_you)

    def add_memory(self, memory):
        self.memories.append(memory)

    def recall_memory(self):
        print(self.memories)
        return self.memories

if __name__ == '__main__':

    jian = Lover('Jian', ['music'], '2020-10-20')

    # inherit the function 'list_in_common()' from the SoulMate.
    jian.list_in_common()

    # could replace the function 'say_hi()' from People.
    jian.say_hi()


