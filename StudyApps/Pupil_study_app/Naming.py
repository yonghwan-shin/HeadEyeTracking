import random

tar = [0,1,2,3,4,5,6,7,8]
sub = int(input())
TCblock = []
EPblock = []
env = ['EU', 'EW']
pos = ['PS', 'PW']

def EP_update(sub_number):
    EPblock.append('S' + str(sub_number))
    if (sub_number)%4 == 0:
        EPblock.append(env[0] + '_' + pos[0])
        EPblock.append(env[0] + '_' + pos[1])
        EPblock.append(env[1] + '_' + pos[0])
        EPblock.append(env[1] + '_' + pos[1])
    elif sub_number%4 == 1:
        EPblock.append(env[0] + '_' + pos[1])
        EPblock.append(env[1] + '_' + pos[0])
        EPblock.append(env[1] + '_' + pos[1])
        EPblock.append(env[0] + '_' + pos[0])
    elif sub_number%4 == 2:
        EPblock.append(env[1] + '_' + pos[0])
        EPblock.append(env[1] + '_' + pos[1])
        EPblock.append(env[0] + '_' + pos[0])
        EPblock.append(env[0] + '_' + pos[1])
    elif sub_number%4 == 3:
        EPblock.append(env[1] + '_' + pos[1])
        EPblock.append(env[0] + '_' + pos[0])
        EPblock.append(env[0] + '_' + pos[1])
        EPblock.append(env[1] + '_' + pos[0])

random.shuffle(tar)
for i in range(len(tar)):
    TCblock.append('T' + str(tar[i]))
    TCblock.append('C' + str(tar.index(tar[i])))
print(tar)
EP_update(sub)
print(EPblock)
print(TCblock)