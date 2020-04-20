
#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Eigen>

using namespace std;
using namespace Eigen;


extern "C"
{
float gbox1(float x,float y,float z,float p,float q,float t)
{
    float g;
    float r;
    float deltx,delty,deltz;
    deltx=(x-p);delty=(y-q);deltz=(z-t);
    r=sqrt((x-p)*(x-p)+(y-q)*(y-q)+(z-t)*(z-t));
    g=6.67*0.001*(-deltx*log(fabs(r+delty))-delty*log(fabs(r+deltx))+deltz*atanf((deltx*delty)/(deltz*r)));
    return g;
}

float gbox2(float x,float y,float z,float p1,float p2,float q,float t)
{
    float dg;

    dg=gbox1(x,y,z,p2,q,t)-gbox1(x,y,z,p1,q,t);
    return dg;
}

float gbox3(float x,float y,float z,float p1,float p2,float q1,float q2,float t)
{
    float dg;
    dg=gbox2(x,y,z,p1,p2,q2,t)-gbox2(x,y,z,p1,p2,q1,t);
    return dg;
}
float gbox4(float x,float y,float z,float p2,float p1,float q2,float q1,float t2,float t1)//x,y,z.xϣx£yϣy£zϣz£rol
{
    float dg;
    dg=gbox3(x,y,z,p1,p2,q1,q2,t2)-gbox3(x,y,z,p1,p2,q1,q2,t1);
    return dg;
}

float gbox5_va(int deltxn,int deltyn,int deltzn,int xnum,int ynum,int znum,float dx,float dy,float dz,float measureh)
{
    float dg;
    float x,y,z,p2,p1,q2,q1,t2,t1;
    x=0;y=0;z=-measureh-0.5*dz;
    p2=deltxn*dx+0.5*dx;p1=deltxn*dx-0.5*dx;
    q2=deltyn*dy+0.5*dy;q1=deltyn*dy-0.5*dy;
    t2=deltzn*dz+0.5*dz;t1=deltzn*dz-0.5*dz;

    dg=gbox4(x,y,z,p2,p1,q2,q1,t2,t1);
    return dg;
}

void forward_va(float *Va_a,int xnum,int ynum,int znum,float dx,float dy,float dz,float measureh)
{
    int n = 0;
    float temp = 0;

#pragma acc kernels present(Va_a)
{
    #pragma acc loop independent collapse(3)
    for(int vax=1-xnum;vax<xnum;vax++)//vax<=xnum-1
    {
        for(int vay=1-ynum;vay<ynum;vay++)//vay<=ynum-1
        {

            for(int vaz=0;vaz<znum;vaz++)//vaz<=znum-1
            {
                n = vaz + znum * (vay+ynum-1) + (vax + xnum -1)*znum*(2*ynum-1);//
                Va_a[n] = gbox5_va(vax,vay,vaz,xnum,ynum,znum,dx,dy,dz,measureh);
            }
        }
    }
}

}


float getAij(int p,int q,float *va,int xnum,int ynum,int znum)
{
    int ix,iy,iz;
    int ik;
    int a,b,c;
    int m,n;

    c=q/(ynum*xnum)+1;
    b=(q%(ynum*xnum))/xnum+1; 
    a=((q%(xnum*ynum))%xnum)+1;
    m=p%xnum+1;
    n=p/xnum+1;

    ix=m-a+xnum;
    iy=n-b+ynum;
    iz=c;

    ik=(ix-1)*(2*ynum-1)*znum+(iy-1)*znum+iz-1;
    return va[ik];
}

void AplusS(float *Va_a,float *S_a,float *abn_a,int xnum,int ynum,int znum)
{
    int Acols=xnum*ynum*znum;
    int Arows=xnum*ynum;
    //TODO: add parallel
#pragma acc kernels present(Va_a,S_a,abn_a)
{
    #pragma acc loop independent
    for(int i=0;i<Arows;i++)
    {
        float value=0;
        for(int m=0;m<Acols;m++)
        {
            value=value+getAij(i,m,Va_a,xnum,ynum,znum)*S_a[m];
        }
        abn_a[i]=value;
    }
}
}


int test_function(int a)
{
    return 2*a;
}


}
