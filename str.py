import pickle
import numpy as np
import streamlit as st
load_model=pickle.load(open('bc.sav','rb'))


a=int(input("Enter your Radius Mean"))
b=int(input("\nEnter your Texture Mean"))
c=int(input("\nEnter your Perimeter Mean"))
d=int(input("\nEnter your Area Mean"))
e=int(input("\nEnter your Smoothness"))
f=int(input("\nEnter your Compactness Mean"))
g=int(input("\nEnter your Concavity Mean"))
h=int(input("\nEnter your Concave points mean"))
i=int(input("\nEnter your Symmetry Mean"))
j=int(input("\nEnter your fractal Dimension mean"))
k=int(input("\nEnter your Radius"))
l=int(input("\nEnter your Texture "))
m=int(input("\nEnter your Perimeter"))
n=int(input("\nEnter your Area"))
o=int(input("\nEnter your Smoothness"))
p=int(input("\nEnter your Compactness"))
q=int(input("\nEnter your Concavity"))
r=int(input("\nEnter your Concave points"))
s=int(input("\nEnter your symmetry"))
t=int(input("\nEnter your fractal dimension"))
u=int(input("\nEnter your radius worst"))
v=int(input("\nEnter your Texture worst"))
x=int(input("\nEnter your Perimeter Worst"))
y=int(input("\nEnter your area worst"))
z=int(input("\nEnter your smoothness worst"))
aa=int(input("\nEnter your Compactness Worst"))
bb=int(input("\nEnter your concavity_worst"))
cc=int(input("\nEnter your concave points_worst"))
dd=int(input("\nEnter your symmetry_worst"))
ee=int(input("\nEnter your fractal dimension worst"))




cd=np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,x,y,z,aa,bb,cc,dd,ee]])
print(cd)
d=load_model.predict(cd)
#print(d)
#print(data['diabetes'][d])

print("-------------------------------------------------------------------------")
if(d==[1]):
    print("------------------------------------------------------------------------")
    print("Malignant")
    print("------------------------------------------------------------------------")
    
else:
    print("------------------------------------------------------------------------")
    print("Benign")
    print("------------------------------------------------------------------------")

print("Thank You")