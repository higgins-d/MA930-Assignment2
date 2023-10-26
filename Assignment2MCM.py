import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime as dt

#We set the seed for the file for reproducibility.

np.random.seed(500000)

#And import the csv file

df = pd.read_csv(r"austin_311_service_requests.csv")

#We remove rows where the 'created_date' (i.e. the call time) is blank, and convert the appropriate strings to floats

df.dropna(subset=['created_date'], inplace=True)
df['open_time_num'] = df['open_time_num'].astype(float)


#And sort the dataframe by created_time

df.sort_values(by=['open_time_num'])

# We remove rows that do not take place in the first 2 months of 2014. These rows will have a open_time_num less than 41640 or greater than or equal to 41700

df_new = df.query('open_time_num > 41640')
df_fin = df_new.query('open_time_num <= 41700')

# We now calculate the time between the call and the start time, add noise, and find the difference in time between each call (the holding time).

times = df_fin['open_time_num'].tolist()

size = len(times)

call_times = [0]*size
for i in range(0, size):
    call_times[i] = times[i] + st.uniform.rvs(loc = -0.005, scale = 0.01)

call_times = np.sort(call_times)
hold_times = []
for i in range(1, size):
    hold_times.append(call_times[i] - call_times[i-1])

#Here, we perform MCMC on the likelihood function of our poisson prior for each lambda.

iterations = 100
logl_i = [0]*iterations
logl_c = [0]*iterations
lambdas = [0]*iterations
lam = 200
for l in range(0, iterations):
    lambdas[l] = lam
    cand = lam + st.uniform.rvs(loc = -4, scale = 8)
    prob_i = [0]*size
    prob_c = [0]*size
    for i in range(0, size-1):
        t = hold_times[i]
        prob_i[i] = np.log(st.poisson.pmf(k = 1, mu = lam*t))
        prob_c[i] = np.log(st.poisson.pmf(k = 1, mu = cand*t))
    logl_i[l] = np.sum(prob_i)
    logl_c[l] = np.sum(prob_c)
    if logl_c[l] < logl_i[l]:
        switch = st.bernoulli(logl_c[l]/logl_i[l])
        if switch == 1:
            lam = cand
        else:
            cand = cand
    else:
        lam = cand

#and plot results

logl_i = max(logl_i)/logl_i
plt.plot(logl_i)
plt.xlabel("Number of iterations")
plt.ylabel("Scaled likelihood")
plt.show()

plt.plot(lambdas)
plt.xlabel("Number of iterations")
plt.ylabel("Lambda")
plt.show()

print(cand)

#We can now simulate the the poisson process with the lambda that was found from our MCMC algorithm

simholdtimes = [0]*size
for i in range(0, size):
    if i == 0:
        simholdtimes[i] = st.expon.rvs(scale = 1/cand)
    else:
        simholdtimes[i] = st.expon.rvs(scale = 1/cand) + simholdtimes[i-1]

#And plot results

xlab = np.linspace(0, size, size)
plt.plot(simholdtimes, label = "Simulated calls process")
plt.plot((call_times-41640), label = "Actual calls process")
plt.ylabel("Time (days)")
plt.xlabel("Cumulative Number of Calls")
plt.legend()
plt.show()


