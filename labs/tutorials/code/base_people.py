
# How to define a class
# A Class is like an object constructor,
# or a "blueprint" for creating objects.

class People:
    def __init__(self, name):
        self.name = name # necessary attribute
        self.birthday = None # not necessary

    def say_hi(self):
        # method
        print('%s say HI!' % self.name)

if __name__ == '__main__':
    # instance it
    joy = People('Joy')

    # now, we have an People object.
    # and this object is able to say hi.
    joy.say_hi()

    # We don't know joy's birthday.
    if joy.birthday is None:
        print('We don\'t know about the birthday.')
        joy.birthday = input('Could you tell us? ')

    print('Okay,', joy.name, '\'s birthday at', joy.birthday, '.')


