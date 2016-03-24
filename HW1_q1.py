import csv
import random
import sys
import math
import matplotlib.pyplot as plt
import traceback
import datetime
import threading
import queue
import multiprocessing as mp
#from collections import namedtuple

title = ['vendor_id', 'pickup_time', 'dropoff_time', 'passenger_count', 'trip_distance',
         'x0', 'y0', 'RatecodeID', 'store_and_fwd_flag', 'x1',
         'y1', 'payment_type', 'fare_amount', 'extra', 'mta_tax',
         'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount']
title_len = title.__len__()
pickup_title = ['x0', 'y0']
dropoff_title = ['x1', 'y1']
required_title = pickup_title
# required_title = dropoff_title
separation = '\n-------------'
# row_count = 100000
row_count = -1
# files = ["try.txt", "try2.txt"]
files = ["data10.csv", "data11.csv", "data12.csv"]
# files = ["data11.csv"]
numOfProcess = 1
# num_worker_threads = 4

def normalizeColumn(aColumn: list) -> (float, list):
    """
    Normalize column: (each value - median)
    """
    avg = sum(aColumn) / aColumn.__len__()
    ret = [x - avg for x in aColumn]
    return (avg, ret)


def getRow(aFile: str) -> list:
    hasReadTitle = False
    type = [int, str, str, int, float,
            float, float, int, str, float,
            float, int, float, float, float,
            float, float, float, float]
    # Record = namedtuple("Record", title)
    with open(aFile) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if not hasReadTitle:
                hasReadTitle = True
                continue
            row = list(cast(val) for cast, val in zip(type, row))
            # yield Record._make(row)
            yield row


class ClusteringKMeans:

    def __init__(self, filename, k):
        """
        0. Init
        """
        self.k = k
        self.metadata = {}
        self.iterationCount = 0
        self.sse = 0

        # # Multi-threading support
        # self.q = queue.Queue()
        # self.threads = []
        # for i in range(num_worker_threads):
        #     t = threading.Thread(target=self.assignPointsToClusterMulti)
        #     t.start()
        #     self.threads.append(t)


        """
        1. Read file & put in data
        """
        # self.data = {title[i]: [] for i in range(title_len) if title[i] in required_title}
        self.data = []
        # tmp = {title[i]: [] for i in range(title_len) if title[i] in required_title}
        # # for it in readline(file):
        # it = getRow(filename)
        # for _ in range(row_count):
        #     now = next(it)
        #     for i in range(title_len):
        #         if title[i] in required_title and now[i] != 0.0:
        #             tmp[title[i]].append(now[i])

        for key in row_data:
            self.data.append(row_data[key])

        assert self.data[0].__len__() == self.data[1].__len__()

        self.size = self.data[0].__len__()
        self.belongsTo = [-1 for _ in range(self.size)]

        """
        2. Normalize
        """
        for key, value in enumerate(self.data):
            self.metadata[str(key)+'_avg'], self.data[key] = normalizeColumn(value)

        """
        3. Randomly select k initial centroids
        """
        random.seed()
        self.centroids = [[self.data[i][j] for i in range(required_title.__len__())] for j in random.sample(range(self.size), self.k)]
        # print(self.centroids)

        self.assignPointsToCluster()

    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     # stop workers
    #     for i in range(num_worker_threads):
    #         self.q.put(None)
    #     for t in self.threads:
    #         t.join()

    def updateCentroid(self):
        # start = datetime.datetime.now()
        membersOfCluster = [self.belongsTo.count(i) for i in range(self.k)]
        self.centroids = [[sum(self.data[k][i] for i in range(self.size) if self.belongsTo[i] == centroid)
                           / membersOfCluster[centroid] for k in range(self.data.__len__()) if membersOfCluster[centroid] != 0]
                          for centroid in range(self.k)]
        # end = datetime.datetime.now()
        # print('k = %i: updateCentroid took %s' %(self.k, end-start))

    def assignPointsToCluster(self):
        self.pointsChanged = 0
        self.sse = 0
        # start = datetime.datetime.now()
        self.belongsTo = [self.assignPointToCluster(i) for i in range(self.size)]
        # end = datetime.datetime.now()
        # print('k = %i: assignPointsToCluster took %s' %(self.k, end-start))

        # for i in range(self.size):
        #     self.q.put(i)
        #
        # # block until all tasks are done
        # self.q.join()

    # def assignPointsToClusterMulti(self):
    #     while True:
    #         pointIdx = self.q.get()
    #         if pointIdx is None:
    #             break
    #         self.assignPointToCluster(pointIdx)
    #         self.q.task_done()

    def assignPointToCluster(self, pointIdx: int) -> int:
        min = sys.maxsize
        targetCluster = -1
        for k in range(self.k):
            now = self.distance(pointIdx, k)
            if min > now:
                min = now
                targetCluster = k

        if targetCluster != self.belongsTo[pointIdx]:
            self.pointsChanged += 1

        self.sse += min ** 2

        # self.belongsTo[pointIdx] = targetCluster

        return targetCluster

    def distance(self, pointIdx: int, centroidIdx: int) -> float:
        try:
            sum = 0
            for key, value in enumerate(self.data):
                sum += (self.data[key][pointIdx] - self.centroids[centroidIdx][key]) **2
            return math.sqrt(sum)
        except:
            print('self.data[%i][%i] (size = %i), self.centroids[%i][%i] (size = %i)'
                  %(key, pointIdx, self.data.__len__(), centroidIdx, key, self.centroids.__len__()))
            traceback.print_exc()


    def do(self) -> float:
        isDone = False
        all_start = datetime.datetime.now()
        print('[%i] This is k = %i began' %(all_start.timestamp(), self.k))

        while not isDone:
            iter_start = datetime.datetime.now()

            self.iterationCount += 1
            self.updateCentroid()
            self.assignPointsToCluster()

            iter_end = datetime.datetime.now()
            # if self.iterationCount % 10 == 1:
            #     print('k = %s ==> iteration %ist has done in %s.' %(self.k, self.iterationCount, iter_end-iter_start))

            # Done if fewer than 1% of the points change clusters
            if self.pointsChanged < self.size * 0.01:
                print('')
                isDone = True

        # Show result
        print('==> %i clusters with SSE = %.3f, total %i iterations, duration: %s%s'
              % (self.k, self.sse, self.iterationCount, datetime.datetime.now() - all_start, separation))

        return self.sse

    def showResult(self):
        for centroidIdx, centroid in enumerate(self.centroids):
            print('Centroid #' + str(centroidIdx) + " @ (" + str(centroid[0] + self.metadata['0_avg']) +
                  ", " + str(centroid[1] + self.metadata['1_avg']) + ")")
            for pointIdx in range(self.size):
                if self.belongsTo[pointIdx] == centroidIdx:
                    print('Point #%i: (%f, %f)'
                          %(pointIdx, self.data[0][pointIdx] + self.metadata['0_avg'],
                            self.data[1][pointIdx] + self.metadata['1_avg']))

    def showSummary(self):
        for centroidIdx, centroid in enumerate(self.centroids):
            print('[k = %i] Centroid #%i @(%.14f,%.14f) with r = %.14f [%d points]'
                  %(self.k, centroidIdx, (centroid[0] + self.metadata['0_avg']), (centroid[1] + self.metadata['1_avg']),
                    max(self.distance(pointIdx, centroidIdx) for pointIdx in range(self.size) if self.belongsTo[pointIdx] == centroidIdx),
                    sum(1 for pointIdx in range(self.size) if self.belongsTo[pointIdx] == centroidIdx)))


def doTask(k):
    ret = ClusteringKMeans(file, k)
    sse = ret.do()
    ret.showSummary()
    print('[%i] This is k = %i ... done!' %(datetime.datetime.now().timestamp(),k))
    return sse


def getEachSseByKs(lower: int, upper: int):

    result = {}
    start_time = datetime.datetime.now()
    x_upper = upper
    x_lower = lower
    ret = []

    pool = mp.Pool(numOfProcess)

    try:
        for k in range(x_lower, x_upper+1):
            result[k] = pool.apply_async(doTask, args=(k,))

        for k, aResult in result.items():
            result[k] = ret = aResult.get()
            # print('[%i]This is k = %i ... result get!' %(datetime.datetime.now().timestamp(),k))

    except:
        # print(separation + '\ndata(size = %i):' % ret.data.__len__())
        # print(ret.data)
        print(separation + '\ncentroids(size = %i):' % ret.centroids.__len__())
        print(ret.centroids)
        print(separation + '\nmetadata:')
        print(ret.metadata)

    finally:
        pool.close()
        pool.join()

        result = dict(sorted(result.items()))
        sorted_values = sorted(result.values())
        y_upper = sorted_values[-1]
        y_lower = sorted_values[0]
        time = datetime.datetime.now()-start_time
        print(separation + '\nresult (time: %s):' %time)
        print(result)

        x = list(result.keys())
        y = list(result.values())
        plt.plot(x, y, 'r-x')
        plt.axis([x_lower*0.8, x_upper*1.2, y_lower*0.8, y_upper*1.2])
        for xy in zip(x, y):
            rm = xy[0] % 3
            if rm == 0:
                plt.annotate('%.3f' % xy[1], xy=xy, textcoords='offset points', xytext=(4, 15), arrowprops=dict(arrowstyle='->'))
            elif rm == 1:
                plt.annotate('%.3f' % xy[1], xy=xy, textcoords='offset points', xytext=(0, 40), arrowprops=dict(arrowstyle='->'))
            elif rm == 2:
                plt.annotate('%.3f' % xy[1], xy=xy, textcoords='offset points', xytext=(-30, -30), arrowprops=dict(arrowstyle='->'))
        plt.grid()
        plt.xlabel('k')
        plt.ylabel('SSE')
        fig = plt.gcf()
        fig.canvas.set_window_title('%i-%i_%i-MP%i-pickup' %(row_count, x_lower, x_upper, numOfProcess))
        ax = fig.add_subplot(111)
        file_title = files[0]
        for i in range(1, len(files)):
            file_title += ' & ' + files[i]
        ax.set_title('%s, %i data points, k=%i~%i, elapsed time: %s' %(file_title, row_count, x_lower, x_upper, time))
        plt.show()

        # print(ret.data)
        # print(ret.centroids)
        # print(ret.metadata)

if __name__ == '__main__':

    # Read file
    read_file_start_time = datetime.datetime.now()

    row_data = {title[i]: [] for i in range(title_len) if title[i] in required_title}
    cur_row_count = 0
    for file in files:
        for it in getRow(file):
            if row_count != -1 and cur_row_count >= row_count:
                break
            else:
                cur_row_count += 1

            for i in range(title_len):
                if title[i] in required_title and it[i] != 0.0:
                    row_data[title[i]].append(it[i])
    row_data = dict(sorted(row_data.items()))

    read_file_end_time = datetime.datetime.now()
    print('Finished reading file, total %i lines, taking %s.' %(cur_row_count, (read_file_end_time - read_file_start_time)))

    getEachSseByKs(5, 5)
