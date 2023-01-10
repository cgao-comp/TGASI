class Vertex_C:
    verName = None  
    isObserver = False  
    time = -1  
    infe = False  
    recovery = False  
    degree = -1  
    infected_p = -1.0  
    nextNode = None  

    def __init__(self):
        self.verName = None
        self.isObserver = False
        self.time = -1
        self.infe = False
        self.recovery = False
        self.degree = -1
        self.infected_p = -1.0
        self.nextNode = None
