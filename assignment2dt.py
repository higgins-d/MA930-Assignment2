import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
import pandas as pd
from datetime import datetime as dt

#We set the seed for the file for reproducibility.

np.random.seed(500001)

#And import the csv file

df = pd.read_csv(r"austin_311_service_requests.csv")

#We remove rows where the 'created_date' (i.e. the call time) is blank, and convert the appropriate strings to floats. Note that astype(np.int64) gives a value in seconds
#So we convert to days.

df.dropna(subset=['created_date'], inplace=True)
df['created_date'] = pd.to_datetime(df['created_date'], infer_datetime_format = True).astype(np.int64)

# And sort the dataframe by created_data

df.sort_values(by=['created_date'])

# We remove calls that do not take place in the first 2 months of 2014.

rangestart = pd.Timestamp("01.01.2014 00:00:00").value
rangeend = pd.Timestamp("03.01.2014 00:00:00").value

df_new = df.query(f'created_date > {rangestart}')
df_fin = df_new.query(f'created_date <= {rangeend}')

# We extract the time data and convert it into days. The times list will then be the number of days after 01/01/2014 00:00:00 that a call took place

times = df_fin['created_date'].tolist()
for i in range(0, len(times)):
    times[i] = (times[i]-rangestart)/(24*60*60*1000000000)

# We now find the holding times between calls

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
        switch = st.bernoulli(np.exp(logl_c[l]-logl_i[l]))
        if switch == 1:
            lam = cand
        else:
            cand = cand
    else:
        lam = cand

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
plt.plot((call_times), label = "Actual calls process")
plt.ylabel("Time (days)")
plt.xlabel("Cumulative Number of Calls")
plt.legend()
plt.show()

