
# Import a module
# from <File Name> import <Class Name> or <Function Name>
from base_people import People

# Bcz the definition of SoulMate "soulmate is a person".
# Soulmate will inherit the properties and methods from the People.

class SoulMate(People):
    def __init__(self, name, in_common):
        # By using the super() function, you do not have to use the name of the parent element,
        # it will automatically inherit the methods and properties from its parent

        super(SoulMate, self).__init__(name)
        self.in_common = in_common

    def list_in_common(self):
        print('%s and you have many things in common, like:' % self.name)
        for item in self.in_common:
            print(item)
        return self.in_common

    @staticmethod
    def definition():
        print('Definition of Soulmate: ')
        print('A soulmate is a person with whom one has a feeling of deep or natural affinity.'
             ' This may involve similarity, love, comfort and trust.')


if __name__ == '__main__':
    byul = SoulMate('byul', ['foods', 'cool', 'experience'])
    byul.list_in_common()

    # inherit the methods 'say_hi()' from the People.
    byul.say_hi()

    # inherit the attribute 'birthday' from the People
    byul.birthday = '1997-12-20'
    print(byul.birthday)



