from threading import Thread

class FillGradientsWorker(Thread):
    def __init__(self, graph, node, lo, hi, cutoff):
        Thread.__init__(self)
        self.graph = graph
        self.node = node
        self.lo = lo
        self.hi = hi
        self.mid = lo + (hi - lo) / 2
        self.cutoff = cutoff

    def run(self):
        if(self.hi - self.lo <= self.cutoff):
            self.graph.fill_gradients(self.node, self.lo, self.hi)
            return
        else:
            # divide
            left = FillGradientsWorker(self.graph, self.node, self.lo, self.mid, self.cutoff)
            right = FillGradientsWorker(self.graph, self.node, self.mid, self.hi, self.cutoff)
            left.start()
            right.run()
            left.join()
            return
