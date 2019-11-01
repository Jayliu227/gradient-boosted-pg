import os
import termplotlib as tpl   
# pip install termplotlib --user

while(True):
    name = input('file name is: ')
    file_name = 'reward_records_%s.txt' % name

    if name == 'quit':
        break

    if not os.path.exists(file_name):
        print('file name invalid.')
        continue
    
    f = open(file_name, 'r')
    contents = f.read()
    y = [float(i) for i in contents.split('\n')[:-1]]
    x = range(len(y))
    
    fig = tpl.figure()
    fig.plot(x, y, width=60, height=20)
    fig.show()

