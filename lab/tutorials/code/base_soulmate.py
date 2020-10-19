

class SoulMate():
    def __init__(self, name, in_common=[]):
        self.name = name
        self.birthday = None
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
    # static method no need to instantiate the object.
    # for example: definition()
    SoulMate.definition()

    # list_in_common need to.
    byul = SoulMate('byul', ['foods', 'cool', 'experience'])
    byul.list_in_common()


