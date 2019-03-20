# equacion G(s) = C*(s*I - A)*B
# A**(-1) = Adj(A) / det(a)

#******time domain****#
#x'(t) = Ax(t) + Bx(t)
#y(t)  = Cx(t) +Dx(t)

import matplotlib.pyplot as plt 
import numpy as np
import control as con




r= 1   #10e3 #value resistor 100K ,capacitor= 10uF and inductor
c = 1  #10e-6
l = 1    #e-2


a = np.array([[ 0,1],[-1/(l*c),-r/l]])   #state matrix
b = np.array([[0],[1/l]])                  #input matrix
c = np.array([1/c,0])                     #output matrix
d = np.array([0])                       #action matrix



#take the matrix in state space and getting a function transfer

state_space= con.ss(a,b,c,d)    #state space 
[num,den] = con.tfdata(state_space)
t_function = con.tf(num,den)


#linear simulation  we can see a response to input of system 
#we need u(t)= input of system and x(t) inicial condition
# u =! 0 and x == 0 homogeny solucion # u == 0 and x =! 0 forced solution 
# u =! 0 and x =! 0 full solution


#plot response

sys_state = con.ss(a,b,c,d)
time_simulation = np.arange(0,30,0.01 ,dtype=float)  #time start 0 
input_u = 2 *np.ones(np.size(time_simulation))       #input of 2 volts
start_condition = np.array([[1],[0]])
t,y_out,x_out = con.forced_response(sys_state,time_simulation,input_u,start_condition)
[wn,zetas,poles]= con.damp(sys_state)
#t,y_out, = con.initial_response(sys_state,time_simulation,start_condition)


[time2,yout2] = con.step_response(t_function,time_simulation,start_condition)
plt.figure(1)
plt.plot(time2,yout2,'r')
plt.xlabel('time')
plt.ylabel('volts')



plt.figure(2)
plt.plot(t,y_out,'b*',t,input_u,'g-')  #ploting time x output and time x input
plt.xlabel('time')
plt.ylabel('volts')
plt.show()



print("natural frequency = ",wn)
print("damping constant = ",zetas)
print("system poles = ",poles)




#take the transfer function and now getting an space state


[a,b,c,d]=con.ssdata(t_function)

state_space_2 = con.ss(a,b,c,d) 

my_sys = con.minreal(state_space_2)






print("G(s) = ",t_function)  #function transfer
print("###################")
print(my_sys)      # state space







