
# Define a variable in Python
my_name = 'Joy'
myName = 'Joy'
IAM_RELLY_COOL = True

# How to Define a function
def do_something_really_cool_like_joy():
    # You could do something in the function.
    # for example, like print string.
    print('OK, Maybe ...')
    print('Study Machine Learning hard ? ')
    # ok, maybe this example not cool.

def doNothing():
    # and you also could return something in the end of the function.
    # here return None bcz the function name call `do nothing` haha.
    return None

# If condition
if IAM_RELLY_COOL:
    do_something_really_cool_like_joy()

elif not IAM_RELLY_COOL:
    do_what = doNothing()

else:
    pass

# For-loop a range
for i in range(10):
    print('No.', i, 'time in this for-loop.')

# For-loop a `list`
artist_joy_like = ['deca joins', '9m88', 'Leo Wang', 'Ryan Beatty',
                  'MAMAMOO', 'Soft Lipa', 'Waa Wei', 'The Black Skirts']

# okay, so if you also like these singer, contact me. (X)
for artist in artist_joy_like:
    print(artist)

# For-loop in one-line
odd_numbers = [n for n in range(20) if n %2 == 1]
print(odd_numbers)

# Equal to
odd_numbers = []
for n in range(20):
    if n % 2 == 1:
        odd_numbers.append(n)
print(odd_numbers)




