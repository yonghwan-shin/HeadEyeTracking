import random
import time

targets = [0, 1, 2, 3, 4, 5, 6, 7, 8]
# sub = int(input())
TCblock = []
# EPblock = []
env = ['EU', 'EW']
pos = ['PS', 'PW']
rep = 5


def EP_update(sub_number):
    EPblock = []
    # EPblock.append('S' + str(sub_number))
    if (sub_number) % 4 == 0:
        EPblock.append(env[0] + '_' + pos[0])
        EPblock.append(env[0] + '_' + pos[1])
        EPblock.append(env[1] + '_' + pos[0])
        EPblock.append(env[1] + '_' + pos[1])
    elif sub_number % 4 == 1:
        EPblock.append(env[0] + '_' + pos[1])
        EPblock.append(env[1] + '_' + pos[0])
        EPblock.append(env[1] + '_' + pos[1])
        EPblock.append(env[0] + '_' + pos[0])
    elif sub_number % 4 == 2:
        EPblock.append(env[1] + '_' + pos[0])
        EPblock.append(env[1] + '_' + pos[1])
        EPblock.append(env[0] + '_' + pos[0])
        EPblock.append(env[0] + '_' + pos[1])
    elif sub_number % 4 == 3:
        EPblock.append(env[1] + '_' + pos[1])
        EPblock.append(env[0] + '_' + pos[0])
        EPblock.append(env[0] + '_' + pos[1])
        EPblock.append(env[1] + '_' + pos[0])
    return EPblock


random.shuffle(targets)
for i in range(len(targets)):
    TCblock.append('T' + str(targets[i]))
    TCblock.append('C' + str(targets.index(targets[i])))

# print(targets)
# EP_update(sub)
# print(EPblock)
# print(TCblock)


def make_file_name(target, env, pos, block, c, sub_num):
    t = time.localtime()
    current_time = time.strftime("%m%d%H%M%S", t)
    # print(current_time)
    output = 'T' + str(target) + '_E' + str(env) + '_P' + str(pos) + '_B' + str(block) + '_C' + str(c) + '_S' + str(
        sub_num) + "_" + current_time
    # output = output.join([target, '_E', env, '_P', pos, '_B', block, '_C', c, '_S', sub_num])
    # print(output)
    return output


def make_experiment_array(sub_num):
    EPblock = EP_update(sub_num)
    total_array = []

    for block in range(len(EPblock)):
        for repetition in range(rep):
            random.shuffle(targets)
            for c in range(len(targets)):
                sendstring = make_file_name(target=targets[c], env=EPblock[block][1:2], pos=EPblock[block][3:4],
                                            block=block, c=c, sub_num=sub_num)
                total_array.append(total_array)
                # print(sendstring)
        # end of one block
        # send "BREAK"
        if block == len(EPblock)-1:
            # end of experiment
            # send "FINISH"
            total_array.append("FINISH")
        else:
            total_array.append("BREAK")


    return total_array

exp = make_experiment_array(1)
print(exp[-1])
print(len(exp))