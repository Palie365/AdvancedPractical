import numpy as np
import numpy.random as rnd
import math
import copy


class Data(object):
    def __init__(self):
        self.inbound = np.array([
            10, 20, 26, 26, 28, 33, 24, 25, 26, 32, 18, 21, 28, 26, 8, 12, 4, 4,
            8, 2, 1, 3, 6, 8, 10, 20, 27, 39, 31, 28, 34, 30, 23, 29, 23, 26, 17, 20, 11, 22, 9, 4,
            8, 1, 1, 3, 3, 5, 6, 22, 27, 25, 33, 29, 22, 20, 24, 14, 23, 17, 15, 10, 7, 8, 5, 5,
            12, 6, 2, 7, 1, 3, 11, 21, 25, 26, 30, 35, 34, 40, 28, 35, 28, 21, 15, 13, 19, 14, 13, 6,
            6, 4, 3, 4, 6, 8, 11, 22, 30, 32, 35, 39, 38, 34, 26, 28, 39, 40, 23, 18, 15, 12, 8, 2,
            6, 1, 2, 1, 1, 5, 12, 22, 26, 33, 31, 31, 33, 24, 20, 23, 17, 16, 13, 13, 10, 7, 9, 10,
            5, 1, 1, 1, 2, 2, 6, 6, 16, 21, 17, 14, 21, 18, 16, 18, 19, 15, 9, 5, 5, 6, 7, 5, 5, 1, 2, 5, 4, 2
        ])
        self.outbound = np.array([4, 5, 8, 2, 2, 6, 4, 6, 3, 4, 6, 5, 1, 3, 3, 3, 2, 1,
                                  1, 1, 1, 2, 1, 1, 1, 1, 3, 3, 3, 2, 4, 10, 10, 9, 16, 24, 14, 21, 17, 26, 28, 30,
                                  25, 15, 20, 13, 4, 1, 2, 8, 6, 5, 6, 2, 5, 10, 14, 18, 23, 29, 22, 29, 20, 26, 25, 17,
                                  49, 53, 44, 33, 14, 13, 7, 20, 29, 28, 21, 21, 19, 17, 24, 22, 29, 35, 24, 35, 42, 44,
                                  32, 51,
                                  59, 53, 29, 18, 11, 8, 11, 12, 30, 11, 9, 14, 12, 15, 15, 19, 26, 24, 27, 25, 30, 51,
                                  35, 42,
                                  29, 43, 18, 14, 12, 6, 9, 16, 8, 5, 12, 8, 19, 11, 14, 32, 38, 40, 35, 55, 63, 70, 54,
                                  51,
                                  59, 83, 73, 64, 44, 31, 24, 24, 23, 18, 19, 22, 25, 28, 10, 11, 9, 9, 9, 7, 5, 14, 12,
                                  9, 2, 7, 6, 2, 3, 6,
                                  ])

    def getInboundRate(self):
        return self.inbound

    def getOutboundRate(self):
        return self.outbound


################################### classes ########################
# customer has identification number , and
# arrival time , and service time as a 3-tuple info
class Customer(object):
    def __init__(self, _idnr, _arrtime=np.inf, _sertime=np.inf, _weight=5.0):
        self.info = (_idnr, _arrtime, _sertime, _weight)

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
        self.line = _line

    # adds a customer at the end of the queue
    def addCustomer(self, cust):
        if len(self.line) == 0:
            self.line = [cust]
        else:
            self.line.append(cust)

    # returns and deletes customer with idnr i
    def deleteCustomer(self, i):
        cust = next(cus for cus in self.line if cus.getIdnr() == i)  # find customer i
        self.line.remove(cust)
        return cust

    # gets the customer in front and removes from queue
    def getFirstCustomer(self):
        cust = self.line.pop(0)  # pop(0) removes and returns the first element of the list.
        return cust

    def getLen(self):
        return len(self.line)

    def getLast(self):
        return self.line[-1].getIdnr()


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

    def removeArrival(self):
        for event in self.elist:
            if event.getType() == 'arr':
                self.elist.remove(event)


# state is queue length and number of busy servers
class State(object):
    def __init__(self, _queueLength=0, _numBusy=0):
        self.queueLength = _queueLength
        self.numBusy = _numBusy

    # ############## random stuff ##########################
    # exponential variate
    # using the random number generator rng ( defined in main )


# counter variables per run
# queueArea = int Q(t)dt where Q(t)= length of queue at time t
# waitingTime = sum W_k where W_k= waiting time of k-th customer
# numArr = number of arrivals
# numServed = number served
# numOntime = number of customers startng their service on time
# numWait = number of arrivals who go into queue upon arrival
class QPerformanceMeasures(object):
    def __init__(self, _queueArea=0.0, _waitingTime=0,
                 _numArr=0, _numServed=0, _numOntime=0, _numWait=0, _lateWeight=0.0):
        self.queueArea = _queueArea
        self.waitingTime = _waitingTime
        self.numArr = _numArr
        self.numServed = _numServed
        self.numOntime = _numOntime
        self.numWait = _numWait
        self.lateWeight = _lateWeight

    def addObs(self, obs):
        self.queueArea += obs.queueArea
        self.waitingTime += obs.waitingTime
        self.numArr += obs.numArr
        self.numServed += obs.numServed
        self.numOntime += obs.numOntime
        self.numWait += obs.numWait
        self.lateWeight += obs.lateWeight


#################################################################################

def triangular(weight):
    return (5 + 5.5 * weight) / 60


def ranpoisson(rate):
    return rnd.exponential(1 / rate)

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

def simrun(lamda, c, tc, T, AWT, _Q):
    nc = 0  # customer number = number of arrivals

    Q = _Q
    L = Eventlist()
    X = State(_queueLength=Q.getLen())
    obs = QPerformanceMeasures()

    evt = Event(tc + ranpoisson(lamda[math.floor(tc % 168)]), 'arr', 0)  # first arrival
    L.addEvent(evt)

    while tc < T:

        evt = L.getFirstEvent()  # next event
        te = evt.getTime()
        tp = evt.getType()

        ti = te - tc
        obs.queueArea += ti * X.queueLength  # all the customers in the queue has to wait t_i

        if tp == 'arr':  # arrival event
            nc = HandleArrival(lamda[math.floor(tc % 168)], c, te, nc, X, Q, L, obs)
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
def simulations(lamda_in, lamda_out, lamda_crit, N, tc, T, AWT, n, cost_per_hour, cost_per_hour_critical, shift_length):
    cumobs = QPerformanceMeasures()
    solution = []
    for _ in range(168//shift_length):
        for _ in range(T//168 + 1):
            solution.append([])

    c_inbound = 0
    c_outbound = 0
    c_critical = 0

    crit_queue = [Queue()]
    in_queue = [Queue()]
    out_queue = [Queue()]

    for i in range((T-tc) // shift_length):
        adjustment = (i*shift_length)//168
        start = i * shift_length+tc - adjustment*168
        stop = start + shift_length

        print(f"Shift {i + 1} : {start}-{stop}")
        print("previous crit queuelength = ", crit_queue[i].getLen())
        print("previous queuelength = ", in_queue[i].getLen())
        print("previous queuelength = ", out_queue[i].getLen())


        crit_cost = 0
        penalty = 0
        for j in range(1, N):
            crit_p = QPerformanceMeasures()
            print(f"critical dock: {j}")
            for k in range(n * 5):
                Q_crit = copy.deepcopy(crit_queue[i])
                obs_crit = simrun(lamda_crit, j, start, stop, AWT, Q_crit)
                crit_p.addObs(obs_crit)

            L_c = crit_p.queueArea / (n*5)
            W_c = crit_p.waitingTime / (n*5)
            PW_c = crit_p.numWait / (n*5)
            SL_c = crit_p.numOntime / (n*5)
            PENALTY_COST = crit_p.lateWeight / (5*n) * 500
            print(f"L: {L_c}, Penalty: {PENALTY_COST}")
            COST_P = PENALTY_COST + shift_length * L_c * cost_per_hour + j * cost_per_hour_critical * 3 * shift_length
            print("total cost: ", COST_P)

            Q1 = Queue()
            if crit_cost != 0 and crit_cost < COST_P and L_c < 1/j:
                for z in range(1, round(L_c)):
                    weight = rnd.triangular(0, 5, 10)
                    ser = triangular(weight)
                    cust = Customer(z, stop%168, ser, weight)
                    Q1.addCustomer(cust)
                crit_queue.append(Q1)
                c_critical = j - 1
                break
            else:
                crit_cost = COST_P
                penalty = PENALTY_COST
        print(f"Chosen critical docks: {c_critical} with cost {crit_cost}")

        pairs = count_divisions(N - c_critical)
        print("Amount of pairs to check: ", len(pairs))

        costs = []
        in_q_len_avg = []
        out_q_len_avg = []

        print("\n")
        for j in range(len(pairs)):
            print(f"pair {j + 1} --- {pairs[j]}")
            i_p = QPerformanceMeasures()
            o_p = QPerformanceMeasures()


            for k in range(n):
                Q_in = copy.deepcopy(in_queue[i])
                Q_out = copy.deepcopy(out_queue[i])

                obs_in = simrun(lamda_in, pairs[j][0], start, stop, AWT, Q_in)
                obs_out = simrun(lamda_out, pairs[j][1], start, stop, AWT, Q_out)

                i_p.addObs(obs_in)
                o_p.addObs(obs_out)

            L_i = i_p.queueArea / n
            W_i = i_p.waitingTime / n
            PW_i = i_p.numWait / n
            SL_i = i_p.numOntime / n

            L_o = o_p.queueArea / n
            W_o = o_p.waitingTime / n
            PW_o = o_p.numWait / n
            SL_o = o_p.numOntime / n

            in_q_len_avg.append(copy.deepcopy(L_i))
            out_q_len_avg.append(copy.deepcopy(L_o))

            print("average in queue length ", L_i)
            print("average out queue length ", L_o)

            COST_I = shift_length * L_i * cost_per_hour + pairs[j][0] * cost_per_hour * 3 * shift_length
            COST_O = shift_length * L_o * cost_per_hour + pairs[j][1] * cost_per_hour * 3 * shift_length
            TOTAL_COST = COST_I + COST_O
            costs.append(TOTAL_COST)
            print(f"{COST_I} + {COST_O} = {TOTAL_COST}\n")

        optimal_cost = min(costs)
        c_inbound = pairs[costs.index(optimal_cost)][0]
        c_outbound = pairs[costs.index(optimal_cost)][1]


        q_len_in = in_q_len_avg[costs.index(optimal_cost)]
        q_len_out = out_q_len_avg[costs.index(optimal_cost)]

        Q2 = Queue()
        Q3 = Queue()
        for z in range(1, round(q_len_in)):
            weight = rnd.triangular(0, 5, 10)
            ser = triangular(weight)
            cust = Customer(z, stop%168, ser, weight)
            Q2.addCustomer(cust)
        in_queue.append(Q2)

        for z in range(1, round(q_len_out)):
            weight = rnd.triangular(0, 5, 10)
            ser = triangular(weight)
            cust = Customer(z, stop%168, ser, weight)
            Q3.addCustomer(cust)
        out_queue.append(Q3)


        print(
            f"{c_inbound} - {c_outbound} - {c_critical} ----- {optimal_cost} + {crit_cost} = {optimal_cost + crit_cost}")
        print("\n\n")
        print(i%(T//shift_length))
        print(i//(T//shift_length))
        id = (i+1)%(168//shift_length)
        solution[i] = [id if id > 0 else (168//shift_length), [c_inbound, c_outbound, c_critical, optimal_cost, crit_cost, penalty, optimal_cost + crit_cost]]

    for solutions in solution:
        if solutions != [] and solutions[0] == 1:
            print("\nWEEK")
        print(solutions)
    return


def count_divisions(n):
    # Initialize a list to store valid pairs
    pairs = []

    # Generate pairs (x1, x2) where both are positive
    for x1 in range(1, n + 1):
        for x2 in range(1, n - x1 + 1):
            pairs.append((x1, x2))

    # Return the count of pairs and the pairs themselves
    return pairs


########## main ########################################################
def main():
    data = Data()
    lambda_inbound = data.getInboundRate()
    lambda_outbound = data.getOutboundRate() * 0.8
    lambda_critical = data.getOutboundRate() * 0.2

    tc = 0
    T = 168
    shift_length = 8

    cost_per_hour = 27
    cost_per_hour_critical = 35
    AWT = 1  # we need to finish servicing in 1 hour
    n = 150  # # number of runs

    simulations(lambda_inbound, lambda_outbound, lambda_critical, 22, tc, T, AWT, n, cost_per_hour,
                cost_per_hour_critical, shift_length)


if __name__ == '__main__':
    main()
