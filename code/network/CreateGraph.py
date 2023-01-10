import network.Graph as GraphPackage
import network.Vertex as VertexPackage


class CreateGraph:
    @staticmethod
    def initGraph(network=GraphPackage.Graph_C, file_path="", verNum=0, edgeNum=0):
        network.verNum = verNum
        network.edgeNum = edgeNum
        for index in range(network.verNum):
            vertex = VertexPackage.Vertex_C()
            vertex.verName = str(index + 1)
            vertex.nextNode = None
            vertex.time = -1
            vertex.infe = False
            vertex.recovery = False
            vertex.degree = 0
            network.vertexArray.append(vertex)

        file = open(file_path)
        while 1:
            line = file.readline()

            if not line:
                break
            cut_twoPart = line.split(" ")
            edge0_name = cut_twoPart[0]
            edge1_name = cut_twoPart[1]

            v1 = network.vertexArray[int(edge0_name)]
            if (v1 == None):
                print("输入边存在图中没有的顶点！")
            # 下面代码是图构建的核心：链表操作
            v2 = VertexPackage.Vertex_C()
            v2.verName = edge1_name
            v2.nextNode = v1.nextNode
            v1.nextNode = v2
            v1.degree = v1.degree + 1

            reV2 = network.vertexArray[int(edge1_name)]
            if (reV2 == None):
                print("输入边存在图中没有的顶点！")
            reV1 = VertexPackage.Vertex_C()
            reV1.verName = edge0_name
            reV1.nextNode = reV2.nextNode
            reV2.nextNode = reV1
            reV2.degree = reV2.degree + 1

    @staticmethod
    def resetGraph(network=GraphPackage.Graph_C):
        for index in range(network.verNum):
            v = network.vertexArray[index]
            v.isObserver = False
            v.time = -1
            v.infe = False
            v.recovery = False
            v.infected_p = -1.0

    @staticmethod
    def outputGraph(network=GraphPackage.Graph_C):
        print("输出网络的邻接表为：")
        for index in range(network.verNum):
            v = network.vertexArray[index]
            print(v.verName, end='')
            # print(" ( degree", + %d, + " time", + %d, + " infe", + %d, +")"  % (v.degree, v.time, v.infe)  )
            print(" ( degree", v.degree, " time", v.time, " infe", v.infe, ")", end='')
            current = v.nextNode
            while current != None:
                print("-->%d" % (int(current.verName)), end='')
                current = current.nextNode
            print()

    @staticmethod
    def toMatrix(network=GraphPackage.Graph_C):
        M = [[0 for col in range(network.verNum)] for row in range(network.verNum)]
        for index in range(network.verNum):
            v = network.vertexArray[index]
            current = v.nextNode
            while current != None:
                M[index][int(current.verName)] = 1
                current = current.nextNode
        return M

    @staticmethod
    def getAvgDegree(network: GraphPackage.Graph_C):
        sum = 0.0
        for index in range(network.verNum):
            sum = sum + network.vertexArray[index].degree
        return sum / network.verNum

    @staticmethod
    def getNeighs(network: GraphPackage.Graph_C, v):
        neighs = []
        node = network.vertexArray[v]
        node = node.nextNode
        while node is not None:
            neighs.append(int(node.verName))
            node = node.nextNode

        return neighs
