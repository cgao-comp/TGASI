import sys

file = open("./facebook_adjM.txt")
output = open('./facebook_edge_index0.txt', 'w')

while 1:
    line = file.readline()

    if not line:
        file.close()
        output.close()
        print('finish')
        break

    cut_twoPart = line.split(" ")
    edge0_name = cut_twoPart[0]
    edge1_name = cut_twoPart[1]

    edge0_new = int(edge0_name) - 1
    edge1_new = int(edge1_name) - 1
    new_list = str(edge0_new) + ' ' + str(edge1_new) + '\n'
    output.write(new_list)
