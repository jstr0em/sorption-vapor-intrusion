import numpy as np
import matplotlib.pyplot as plt



def theta(p):
    return theta_r + (theta_s-theta_r)/(1+alpha*p**n)**(1-1/n)

def d_theta(p):
    return alpha*(1/n-1)*n*(theta_s-theta_r)*p**(n-1)*(alpha*p**n+1)**(1/n-2)



# theta_s, theta_r, alpha, n
retention_data = {
'sand': [0.38,0.053,3.5,3.2],
'clay': [0.46, 0.098, 1.3, 1.3],
}



z = np.linspace(0,10, 10000)
rho = 1e3
g = 0.82
p = rho*g*z
fig, (ax1, ax2) = plt.subplots(1,2,dpi=300, sharey=True)


for key, val in retention_data.items():
    theta_s = retention_data[key][0]
    theta_r = retention_data[key][1]
    alpha = retention_data[key][2]
    n = retention_data[key][3]

    ax1.plot(theta(p), z, label=key)
    ax2.semilogx(np.abs(d_theta(p)), z)

ax1.legend()
plt.show()
