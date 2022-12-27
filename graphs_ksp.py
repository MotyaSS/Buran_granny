import matplotlib.pyplot as plt
import csv
import math
import numpy as np


def read_csv(value = '',csvf = ''):
    x_list, y_list = [], []
    with open(csvf, "r") as file:
        reader = csv.DictReader(file, delimiter=",")
        for row in reader:
            if (value == 'Mass'):
                x_list += [float(row[value])]
            else:
                x_list += [int(row[value])]
    return x_list

time = read_csv('Time', 'Энергия-Буран_12220325.csv')
mass = read_csv('Mass', 'Энергия-Буран_12220325.csv')
velocity = read_csv('Velocity', 'Энергия-Буран_12220325.csv')



# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(2, 2)


def axis_graph(n,m, x_l, y_l, title):
    axis[n, m].plot(x_l, y_l)
    axis[n, m].set_title(title)


axis_graph(0,0, time, velocity,"Velocity on Time")
axis_graph(0,1, mass, velocity,"Velocity on Mass")
axis_graph(1,0, time, mass, "Mass on Time")

n = 0
for ax in axis.flat:
    n += 1
    if (n == 1):
        ax.set(xlabel='Time, sec', ylabel='Velocity, m/sec')
    if (n == 2):
        ax.set(xlabel='Mass, tons', ylabel='Velocity, m/sec')
    if (n == 3):
        ax.set(xlabel='Time, sec', ylabel='Mass, tons')

plt.show()

# plt.plot(velocity[:50], mass[:50])
# plt.xlabel('Velocity, m/sec')
# plt.ylabel('Mass, tons')
# plt.title('Velocity on Mass(Kerbin)')
# plt.show()
#
v = [float(x) for x in open('out_v (1).txt')]
m = [float(x) for x in open('out_m (1).txt')]
# plt.plot(v, m)
# plt.title('Velocity on Mass(Earth)')
# plt.xlabel('Velocity, m/sec')
# plt.ylabel('Mass, tons')
# plt.show()

fig, axs = plt.subplots(2)


def axs_graph(n, x_l, y_l, title):
    axs[n].plot(x_l, y_l)
    axs[n].set_title(title)


axs_graph(0,v,m, title='Velocity on Mass(Earth)')
axs_graph(1,velocity[:50],mass[:50], title='Velocity on Mass(Kerbin)')
c = 0
for ax in axs:
    if (c==0):
        ax.set(xlabel='Velocity, m/sec', ylabel='Mass, tons')

plt.show()


def landing(h = 2.5*10**5, s = 8.5 * 10**6, g0 = 9.45):
    vx = []
    h_step = h/100
    s_step = s/100
    for i in range(0, 100):
        vx += [s/(2*h/g0)**0.5]
        h -= h_step
        s -= s_step
    return vx



land = landing()
y_l = list(range(1,250_000, 2500))
plt.plot(land, y_l)
plt.title("Landing")
plt.xlabel('Velocity')
plt.ylabel('Height')

plt.show()


def first_space_speed( M = 5.97 * 10**24, R = 6_371_000, h = 250_000, G = 6.67408 * 10**-11):
    v1 = ( G*M/(R+h) )**(1/2)
    return v1


def period( p = np.pi, R = 6650*10**3):
    v = first_space_speed()
    T = 2*p*R/ v
    return T


print(f'{period()} - seconds')


index = [f'Earth - {math.ceil(first_space_speed())} m/sec', f'Kerbin - { math.ceil(first_space_speed(M = 5.2915158 * 10**22 , R = 600_000, h = 250_000, G = 6.67408004 * 10**-11))} m/sec']
values = [math.ceil(first_space_speed()), math.ceil(first_space_speed(M = 5.2915158 * 10**22 , R = 600_000, h = 250_000, G = 6.67408004 * 10**-11))]
plt.bar(index,values)
plt.title('The first cosmic velocity')
plt.ylabel("m/sec")
plt.show()


values = ['Earth - 2375 tons', 'Kerbin - 719.7 tons']
val = [2375, 719.7]
plt.bar(values, val)
plt.title('Starter Mass')
plt.show()


values = [f'Earth - {94} minutes ({94*60} sec)', f'Kerbin - {math.ceil(max(time)/60)} minutes ({max(time)} sec)']
val = [94, math.ceil(max(time)/60)]
plt.bar(values, val)
plt.title("Flight duration")
plt.show()


def Tsiolkovsky_formula(g0 = 9.8, i = 308.5, m_start = 2375, m_fin = 428.857):
    v_max = i * g0 * math.log1p(m_start/m_fin)
    return v_max


def curr_weight_2(m_k = 72.557, v = (7753.389)/2, g0 = 9.81, i = 453.2):
    m_t = (math.exp(v/(g0*i)) - 1) * m_k
    return m_t + m_k

def curr_weight_general(m_k = 262.4 + curr_weight_2() , v = (7753.3898)/2, g0 = 9.81, i = 308.5):
    m_t = (math.exp(v/(g0*i)) - 1) * m_k
    return m_t

print(f'Скорость, которую нужно достичь для выхода на орбиту земли: {math.ceil(first_space_speed())} м/с')
print(f'Масса топлива, которая нужна энергии(и бурану): {math.ceil(curr_weight_general(v = first_space_speed()))} тонн')
print(f'Скорость, которую разовьем с такой массой: {math.ceil(Tsiolkovsky_formula(m_start= 429+math.ceil(curr_weight_general())))} м/с')

print(f'Первая космическая для Кербина: {math.ceil(first_space_speed(M = 5.2915158 * 10**22 , R = 600_000, h = 250_000, G = 6.67408004 * 10**-11))}')