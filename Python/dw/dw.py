import math
tp_list=[
        [1.32,1.20,1.14,1.11,1.09,1.08,1.07,1.06,1.04,1.03],
        [2.92,2.35,2.13,2.02,1.94,1.86,1.83,1.76,1.73,1.71],
        [4.30,3.18,2.78,2.57,2.46,2.37,2.31,2.26,2.15,2.09],
        [9.93,5.84,4.60,4.03,3.71,3.50,3.36,3.25,2.98,2.86]
        ]

kp_list=[
        [1,1.65,1.96,2.58],
        [1.183,1.559,1.645,1.715]
        ]

def tkc(P,N,M):
    if M==0:
        C=3
        t=0
    else:
        C=math.sqrt(3)
        t=1

    if 3<=N and N<=10:
        n=N-3
    elif N==15:
        n=8
    elif N==20:
        n=9
    else:
        return

    if P==0.68:
        p=0
    elif P==0.90:
        p=1
    elif P==0.95:
        p=2
    elif P==0.99:
        p=3
    else:
        return

    return tp_list[p][n],kp_list[t][p],C

def mean(l):
    s=0
    for i in l:
        s+=i
    return s/len(l)

def std_dev(l):
    aver=mean(l)
    s=0
    for i in l:
        s+=(i-aver)**2
    return math.sqrt(s/(len(l)-1))

def d(l,P=0.68,c=0,dB=0):
    tp,kp,C=tkc(P,len(l),c)
    print('tp=   ',tp)
    print ('kp=   ',kp)
    print ('C=    ',C)
    aver=mean(l)
    print ('aver= ',aver)
    stddev=std_dev(l)
    print ('dev=  ',stddev)
    UAD=stddev/math.sqrt(len(l))
    uA=tp*UAD
    UBD=dB/C
    uB=kp*UBD
    print ('UAD=  ',UAD)
    print ('UBD=  ',UBD)
    print ('uA=   ',uA)
    print ('uB=   ',uB)
    u=math.sqrt(uA**2+uB**2)
    print ('u=    ',u)
    print ('P=    ',P)

def help():
    print("d(l,P=0.68,c=0,dB=0)")
    print("r(k,a,u)")
    print("f(x,y)")

def r(k,a,u):
    if(len(a)!=len(u)):
        print("error")
        return
    y=0
    for i,j in zip(a,u):
        y+=(j/i)**2
    y=k*math.sqrt(y)
    print("result=  ",y)
    return y

def f(x,y):
    if(len(x)!=len(y)):
        print("error")
        return 
    x_=sum(x)/len(x)
    y_=sum(y)/len(y)

    xy=[i*j for i,j in zip(x,y)]
    xy_=sum(xy)/len(xy)

    x2=[i*i for i in x]
    x2_=sum(x2)/len(x2)
    y2=[i*i for i in y]
    y2_=sum(y2)/len(y2)

    lxy=len(x)*(xy_-x_*y_)
    lxx=len(x)*(x2_-x_**2)
    lyy=len(y)*(y2_-y_**2)

    m=lxy/lxx
    b=y_-m*x_
    r=lxy/math.sqrt(lxx*lyy)
    Sm=m*math.sqrt((1/(r**2)-1)/(len(x)-2))
    Sb=Sm*math.sqrt(x2_)


    print("x_=   ",x_)
    print("y_=   ",y_)
    print("xy_=  ",xy_)
    print("x2_=  ",x2_)
    print("y2_=  ",y2_)
    print('''lxy=len(x)*(xy_-x_*y_)
    lxx=len(x)*(x2_-x_**2)
    lyy=len(y)*(y2_-y_**2)

    m=lxy/lxx
    b=y_-m*x_
    r=lxy/math.sqrt(lxx*lyy)
    Sm=m*math.sqrt((1/(r**2)-1)/(len(x)-2))
    Sb=Sm*math.sqrt(x2_)''')
    print("lxy=  ",lxy)
    print("lxx=  ",lxx)
    print("lyy=  ",lyy)
    print("m=    ",m)
    print("b=    ",b)
    print("r=    ",r)
    print("Sm=   ",Sm)
    print("Sb=   ",Sb)
