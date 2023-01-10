class Vertex_C:
    verName = None  # 节点名称
    isObserver = False  # 节点是否为观察点
    time = -1  # 被感染的真实时间
    infe = False  # 该节点是否被感染
    recovery = False  # 该节点是否回复
    degree = -1  # 当前节点的度
    infected_p = -1.0  # 感染率
    nextNode = None  # 链表元素

    def __init__(self):
        self.verName = None
        self.isObserver = False
        self.time = -1
        self.infe = False
        self.recovery = False
        self.degree = -1
        self.infected_p = -1.0
        self.nextNode = None
