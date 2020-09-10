import random
import time

targets = [0, 1, 2, 3, 4, 5, 6, 7]

env = ['EU', 'EW']
pos = ['PS', 'PW']
rep = 5

# DEBUG - short trials
# targets = [0, 1]
# rep =2

def EP_update(sub_number):
    EPblock = []
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
    # print(EPblock)
    return EPblock

def make_file_name(target, env, pos, block, c, sub_num):
    output = 'T' + str(target) + '_E' + str(env) + '_P' + str(pos) + '_B' + str(block) + '_C' + str(c) + '_S' + str(
        sub_num)
    return output

def make_experiment_array_walkonly(sub_num):
    EPblock = ['U','W'] if sub_num %2 ==0 else ['W','U']
    total_array = []
    for block in range(len(EPblock)):
        for repetition in range(rep):
            random.shuffle(targets)
            for c in range(len(targets)):
                sendstring = 'T' + str(targets[c]) + '_E' + str(EPblock[block]) + '_B' + str(repetition) + '_C' + str(c) + '_S' + str(sub_num)
                total_array.append(sendstring)
        if block == len(EPblock) -1 :
            total_array.append("FINISH")
        else:
            total_array.append("BREAK")
    total_array.reverse()
    print('Total Trials:',len(total_array))
    # print(total_array)
    return total_array
def make_experiment_array(sub_num):
    EPblock = EP_update(sub_num)
    total_array = []

    for block in range(len(EPblock)):
        for repetition in range(rep):
            random.shuffle(targets)
            for c in range(len(targets)):
                sendstring = make_file_name(target=targets[c], env=EPblock[block][1:2], pos=EPblock[block][4:5],
                                            block=repetition, c=c, sub_num=sub_num)
                total_array.append(sendstring)
        # end of one block
        # send "BREAK"
        if block == len(EPblock)-1:
            # end of experiment
            # send "FINISH"
            total_array.append("FINISH")
        else:
            total_array.append("BREAK")
    total_array.reverse()


    # for _ in range(123):
    #     total_array.pop()
    print(len(total_array))
    print(total_array)
    return total_array



def current_add(title):
    t = time.localtime()
    current_time = time.strftime("%m%d%H%M%S", t)
    output = title + "_" + current_time
    return output

# exp = make_experiment_array(1)
# print(exp)
# print(exp[-1])
# print(len(exp))

if __name__ =="__main__":
    tt = make_experiment_array_walkonly(1)
    # print(tt)
    # print(len(tt))