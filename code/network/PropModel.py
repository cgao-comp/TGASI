import network.Graph as GraphPKG
import network.Vertex as VPKG
import random


class SI_C:
    infected_p = -1.0
    timestamp = -1
    max_timestamp = 10000
    infection_set = []
    source = -1

    def SI_Prop(self, network=GraphPKG.Graph_C, infected_p=-1, ppt_of_I=-1, define_source=-1):
        self.init_paras()
        if define_source == -1:
            self.infection_set.append(random.randint(1, network.verNum))
        else:
            self.infection_set.append(define_source)
        self.source = self.infection_set[0]
        for self.timestamp in range(1, self.max_timestamp):
            infection_set_unchanged = self.infection_set.copy()
            for infection_person_typeInt in infection_set_unchanged:
                infection_vertex = network.vertexArray[infection_person_typeInt - 1]
                neighbor = infection_vertex.nextNode
                while neighbor != None:
                    if not network.vertexArray[int(neighbor.verName) - 1].infe:
                        # 执行概率公式，概率成功则为感染
                        if random.random() <= infected_p:
                            self.infection_set.append(int(neighbor.verName))
                            v = network.vertexArray[int(neighbor.verName) - 1]
                            v.infe = True
                            v.time = self.timestamp

                            if len(self.infection_set) >= float(network.verNum) * ppt_of_I:
                                # print(self.infection_set)
                                # print(len(self.infection_set))
                                return
                    neighbor = neighbor.nextNode

    def getStateLabel(self, network=GraphPKG.Graph_C):
        Y = [-1 for i in range(network.verNum)]
        for index in range(network.verNum):
            v = network.vertexArray[index]
            if(v.infe == True):
                Y[index] = 1
        # print("Y:", Y)
        return Y

    def init_paras(self):
        self.infected_p = -1.0
        self.timestamp = -1
        self.max_timestamp = 10000
        self.infection_set = []
        self.source = -1
