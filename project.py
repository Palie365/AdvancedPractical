import numpy as np
import numpy.random as rnd
from gurobipy import *

class Data(object):
    def __init__(self):
        self.inbound = np.array([
                        [5, 8, 8, 12, 6, 6, 5],
                        [1, 2, 1, 6, 4, 1, 1],
                        [2, 1, 1, 2, 3, 2, 1],
                        [5, 3, 3, 7, 4, 1, 1],
                        [4, 6, 3, 1, 6, 1, 2],
                        [2, 8, 5, 3, 8, 5, 2],
                        [10, 10, 6, 11, 11, 12, 6],
                        [20, 20, 22, 21, 22, 22, 6],
                        [26, 27, 27, 25, 30, 26, 16],
                        [26, 39, 25, 26, 32, 33, 21],
                        [28, 31, 33, 30, 35, 31, 17],
                        [33, 28, 29, 35, 39, 31, 14],
                        [24, 34, 22, 34, 38, 33, 21],
                        [25, 30, 20, 40, 34, 24, 18],
                        [26, 23, 24, 28, 26, 20, 16],
                        [32, 29, 14, 35, 28, 23, 18],
                        [18, 23, 23, 28, 39, 17, 19],
                        [21, 26, 17, 21, 40, 16, 15],
                        [28, 17, 15, 15, 23, 13, 9],
                        [26, 20, 10, 13, 18, 13, 5],
                        [8, 11, 7, 19, 15, 10, 5],
                        [12, 22, 8, 14, 12, 7, 6],
                        [4, 9, 5, 13, 8, 9, 7],
                        [4, 4, 5, 6, 2, 10, 5]
                    ]).flatten(order="F")
        self.outbound = np.array([
                        [2, 1, 25, 49, 59, 29, 59],
                        [7, 1, 15, 53, 53, 43, 83],
                        [6, 1, 20, 44, 29, 18, 73],
                        [2, 2, 13, 33, 18, 14, 64],
                        [3, 1, 4, 14, 11, 12, 44],
                        [6, 1, 1, 13, 8, 6, 31],
                        [4, 1, 2, 7, 11, 9, 24],
                        [5, 1, 8, 20, 12, 16, 24],
                        [8, 3, 6, 29, 30, 8, 23],
                        [2, 3, 5, 28, 11, 5, 18],
                        [2, 3, 6, 21, 9, 12, 19],
                        [6, 2, 2, 21, 14, 8, 22],
                        [4, 4, 5, 19, 12, 19, 25],
                        [6, 10, 10, 17, 15, 11, 28],
                        [3, 10, 14, 24, 15, 14, 10],
                        [4, 9, 18, 22, 19, 32, 11],
                        [6, 16, 23, 29, 26, 38, 9],
                        [5, 24, 29, 35, 24, 40, 9],
                        [1, 14, 22, 24, 27, 35, 9],
                        [3, 21, 29, 35, 25, 55, 7],
                        [3, 17, 20, 42, 30, 63, 5],
                        [3, 26, 26, 44, 51, 70, 14],
                        [2, 28, 25, 32, 35, 54, 12],
                        [1, 30, 17, 51, 42, 51, 9]
                    ]).flatten(order="F")
        
    def getInboundRate(self):
        return self.inbound

    def getOutboundRate(self):
        return self.outbound


"""
    def RatePerShift(self, data):
        shift_size = 8
        sum_last = 0
        sum = 0
        arrival_rates = np.array([])
        k = 0
        for i in range(7):
            for j in range(24):
                if k%8 == 0 and k!=0:
                    arrival_rates = np.append(arrival_rates, sum/shift_size)
                    sum = 0
                
                if i == 0 and j < 6:
                    sum_last += data[j][i]
                elif i == 6 and j > 21:
                    sum_last += data[j][i]
                else:
                    sum += data[j][i]
                    k += 1

        arrival_rates = np.append(arrival_rates, sum_last/shift_size)
        return arrival_rates

"""

 ################################### classes ########################
 # customer has identification number , and
 # arrival time , and service time as a 3-tuple info
class Customer(object):
    def __init__(self, _idnr, _arrtime=np.inf, _sertime=np.inf, _weight=5):
        self.info=(_idnr, _arrtime, _sertime, _weight)

    def getIdnr(self):
        return self.info[0]

    def getArrTime(self):
        return self.info[1]

    def getSerTime(self):
        return self.info[2]
    
    def getWeight(self):
        return self.info[3]
    
######################################################################
 # list of waiting customers
class Queue(object):
    def __init__(self, _line=[]):
        self.line =_line

 # adds a customer at the end of the queue
    def addCustomer(self, cust):
        if len(self.line)==0:
            self.line = [cust]
        else:
            self.line.append(cust)

 # returns and deletes customer with idnr i
    def deleteCustomer(self,i):
        cust = next(cus for cus in self.line if cus.getIdnr()== i)  # find customer i
        self.line.remove(cust)
        return cust

 # gets the customer in front and removes from queue
    def getFirstCustomer(self):
        cust = self.line.pop(0)   #  pop(0) removes and returns the first element of the list.
        return cust


# events are arrivals , and departures from agents
# as 3-tuple info (time of event, type of event, and idnr)
# idnr for arrival is 0
# idnr for departures is customer idnr
class Event(object):
    def __init__(self, _time=np.inf, _type='', _idnr=0):
        self.info = (_time, _type, _idnr)

    def getTime(self):
        return self.info[0]

    def getType(self):
        return self.info[1]

    def getIdnr(self):
        return self.info[2]


##################################################################
# events are ordered chronologically
# always an arrival event
class Eventlist(object):
    def __init__(self, _elist=[]):
        self.elist = _elist

    # adds according to event time
    def addEvent(self, evt):
        if len(self.elist) == 0:
            self.elist = [evt]
        else:
            te = evt.getTime()
            if te > self.elist[-1].getTime():  # add event to the end of the list if its event time is the largest
                self.elist.append(evt)
            else:
                evstar = next(ev for ev in self.elist
                              if ev.getTime() > te)
                evid = self.elist.index(evstar)
                self.elist.insert(evid, evt)  # add event to the right position to ensure the chronological order.

    # returns oldest event and removes from list
    def getFirstEvent(self):
        evt = self.elist.pop(0)
        return evt

    # deletes event as sociated with customer with idnr i
    def deleteEvent(self, i):
        evt = next(ev for ev in self.elist if ev.getIdnr() == i)
        self.elist.remove(evt)


# counter variables per run
# queueArea = int Q(t)dt where Q(t)= length of queue at time t
# waitingTime = sum W_k where W_k= waiting time of k-th customer
# numArr = number of arrivals
# numServed = number served
# numOntime = number of customers startng their service on time
# numWait = number of arrivals who go into queue upon arrival
class QPerformanceMeasures(object):
    def __init__(self, _queueArea=0.0, _waitingTime=0,
                 _numArr=0, _numServed=0, _numOntime=0, _numWait=0, _cost =0 , _lateWeight = 0):
        self.queueArea = _queueArea
        self.waitingTime = _waitingTime
        self.numArr = _numArr
        self.numServed = _numServed
        self.numOntime = _numOntime
        self.numWait = _numWait
        self.cost = _cost
        self.lateWeight = _lateWeight

    def addObs(self, obs):
        self.queueArea += obs.queueArea
        self.waitingTime += obs.waitingTime
        self.numArr += obs.numArr
        self.numServed += obs.numServed
        self.numOntime += obs.numOntime
        self.numWait += obs.numWait
        self.cost += obs.cost
        self.lateWeight += obs.lateWeight


#################################################################################

# state is queue length and number of busy servers
class State(object):
    def __init__(self, _queueLength=0, _numBusy=0):
        self.queueLength = _queueLength
        self.numBusy = _numBusy

    # ############## random stuff ##########################
    # exponential variate
    # using the random number generator rng ( defined in main )

def triangular(weight):
    return (5 + 5.5 * weight)/60

def ranpoisson(rate):
    return rnd.exponential(1/rate)

    # ############# arrival and departure routines #########
    # lamda = arrival rate

    # c = number of servers
    # te = current event time
    # nc = customer number
    # X = state ( queue length and busy servers )
    # Q = state ( queue of customers )
    # L = eventlist
    # obs = vector of counter variables


def HandleArrival(lamda, c, te, nc, X, Q, L, obs):
    nc += 1
    weight = rnd.triangular(0, 5, 10)
    
    ser = triangular(weight)  # required service time

    eva = Event(te + ranpoisson(lamda), 'arr', 0)  # next arrival
    L.addEvent(eva)

    if X.numBusy < c:  # there is a free agent
        X.numBusy += 1
        obs.numServed += 1
        obs.numOntime += 1
        evb = Event(te + ser, 'dep', nc)  # departure from agent
        L.addEvent(evb)
    else:  # all agents busy , customer joins queue
        cust = Customer(nc, te, ser, weight)
        Q.addCustomer(cust)
        X.queueLength += 1
        obs.numWait += 1
    return nc


def HandleDeparture(AWT, te, X, Q, L, obs):
    if X.queueLength == 0:  # no one is waiting
        X.numBusy -= 1
    else:  # takes the first in line
        first = Q.getFirstCustomer()  # delete from queue
        X.queueLength -= 1
        w = te - first.getArrTime()  # waiting time
        obs.waitingTime += w
        if w < AWT:
            obs.numOntime += 1
        else:
            obs.lateWeight += first.getWeight()
        obs.numServed += 1
        i = first.getIdnr()
        ser = first.getSerTime()
        evb = Event(te + ser, 'dep', i)  # departure from agent
        L.addEvent(evb)


# ######### simulations #####################
# simulation run until end of day T
# AWT = accepted waiting time

def simrun(lamda, c, tc, T, AWT, shift_length):
    nc = 0  # customer number = number of arrivals
    shift = 0

    X = State()
    Q = Queue()
    L = Eventlist()
    obs = QPerformanceMeasures()

    evt = Event(ranpoisson(lamda[int((tc-1)//1)]), 'arr', 0)  # first arrival
    L.addEvent(evt)

    while tc < T:
        if (tc + 2) // shift_length > T/shift_length - 1:
            shift = 0
        else:
            shift = (tc + 2) // shift_length

        evt = L.getFirstEvent()  # next event
        te = evt.getTime()
        tp = evt.getType()
        ti = te - tc
        obs.queueArea += ti * X.queueLength  # all the customers in the queue has to wait t_i

        if tp == 'arr':  # arrival event
            nc = HandleArrival(lamda[int((tc-1)//1)], c, te, nc, X, Q, L, obs)
        else:  # departure event
            HandleDeparture(AWT, te, X, Q, L, obs)
            tc = te

    obs.queueArea /= te  # average queue length
    obs.waitingTime /= obs.numServed  # average waiting time
    obs.numArr = nc
    obs.numOntime /= obs.numServed  # fraction served on time
    obs.numWait /= nc  # prob of waiting (delay prob)

    return obs

###########################################################################
# do n runs , collect statistics and report output
def simulations(lamda, N, tc, T, AWT, n, shift_length, cost_per_hour):
    cumobs = QPerformanceMeasures()
    for j in range(n):
        obs = simrun(lamda,  N, tc, T, AWT, shift_length)
        cumobs.addObs(obs)

    L = cumobs.queueArea / n
    W = cumobs.waitingTime / n
    PW = cumobs.numWait / n
    SL = cumobs.numOntime / n
    TOTAL_COST = N*W*cost_per_hour + N*cost_per_hour*3*(T-1)
    
    print('average queue length L :', L)
    print('average waiting time W :', W)
    print('delay probability P_w :', PW)
    print('service level SL :', SL)
    print('Total expected cost:', TOTAL_COST)
    
    return W
    
    
def simulations_critical(lamda, N, tc, T, AWT, n, shift_length, cost_per_hour):
    cumobs = QPerformanceMeasures()
    for j in range(n):
        obs = simrun(lamda, N, tc, T, AWT, shift_length)
        cumobs.addObs(obs)

    L = cumobs.queueArea / n
    W = cumobs.waitingTime / n
    PW = cumobs.numWait / n
    SL = cumobs.numOntime / n
    PENALTY_COST = cumobs.lateWeight / n * 500
    TOTAL_COST = PENALTY_COST + N*W*cost_per_hour + N*cost_per_hour*3*(T-1)
    
    print('average queue length L :', L)
    print('average waiting time W :', W)
    print('delay probability P_w :', PW)
    print('service level SL :', SL)
    
    print('Penalty cost:', PENALTY_COST)
    print('Total expected cost:', TOTAL_COST)

    return W, PENALTY_COST

########## main ########################################################
def main():
    data = Data()
    lambda_inbound = data.getInboundRate()
    lambda_outbound = data.getOutboundRate()*0.8
    lambda_critical = data.getOutboundRate()*0.2

    tc = 0
    shift_length = 8
    
    c_inbound = 10  # number of servers
    c_outbound = 12
    c_critical = 2
    
    cost_per_hour = 27*3
    cost_per_hour_critical = 35*3

    AWT = 1 # we need to finish servicing in 1 hour
    #T = 7*24+1 # run for 7 days + 1 hour as we start at 1
    T = 168
    n = 200  # number of runs

    # print('CRITICAL OUTBOUND')
    # simulations_critical(lambda_critical, c_critical, tc, T, AWT, n, shift_length, cost_per_hour_critical)
    # print('NON-CRITICAL INBOUND')
    # simulations(lambda_inbound, c_inbound, tc, T, AWT, n, shift_length, cost_per_hour)
    # print('NON-CRITICAL OUTBOUND')
    # simulations(lambda_outbound, c_outbound, tc, T, AWT, n, shift_length, cost_per_hour)

    mod = Model('IP1')
    S = mod.addMVar(vtype=GRB.BINARY, shape = (T/shift_length, 22), name='S')
    I = mod.addMVar(vtype=GRB.BINARY, shape = (T/shift_length, 1), name='I')
    C = mod.addMVar(vtype=GRB.BINARY, shape = (T/shift_length, 1), name='C')
    O = mod.addMVar(vtype=GRB.BINARY, shape = (T/shift_length, 1), name='O')
    mod.update()

    mod.setObjective(sum(sum(I[j]*O[j] * S[i][j] * (shift_length*cost_per_hour +
                        simulations(lambda_inbound, c_inbound, i*8-2, i*8-2+8, AWT, n, shift_length, cost_per_hour)
                        * cost_per_hour) for j in range(22)) for i in range(T/shift_length)) +
                     sum(sum(I[j] * O[j] * S[i][j] * (shift_length * cost_per_hour +
                        simulations(lambda_outbound, c_outbound, i*8-2, i*8-2+8, AWT, n, shift_length, cost_per_hour)
                        * cost_per_hour) for j in range(22)) for i in range(T/shift_length))
                                                    )



if __name__ == '__main__':
    main()