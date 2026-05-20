#include "PID.h"
#include <stdlib.h>
#include <stdbool.h>

struct PID{
    //pid系数
    double Kp;
    double Ki;
    double Kd;

    double error;       //此次误差
    double ierror;      //误差积分
    double dvalue;      //值微分

    double nowValue;    //此次的值
    double lastValue;   //上次的值
    double targetValue; //所需值

    double CtlValue;    //输出控制值

    //输出限制
    double CtlMax;
    double CtlMin;

    double dt;//时间间隔

    bool first; //首次调用标识，防止由于首次调用时lastValue与nowValue差别过大造成控制值过大
};

//PID对象创建函数
PID* pid_create(double kp, double ki, double kd, double Ctlmax, double Ctlmin,double dt)
{
    PID* p = (PID*)malloc(sizeof(PID));
    p->Kp = kp;
    p->Ki = ki;
    p->Kd = kd;
    p->lastValue = 0.0;
    p->ierror = 0.0;
    p->CtlMax = Ctlmax;
    p->CtlMin = Ctlmin;
    p->dt = dt;
    p->first = true;
    return p;
}

void pid_clear(PID *in) //重置
{
    in->lastValue = 0.0;
    in->ierror = 0.0;
    in->first = true;
}

double pid_step(PID *in,double value,double target)  //传入当前值和目标值,返回控制值
{
    if(in->first == true)
    {
        in->lastValue = value;
        in->first = false;
    }
    
    in->nowValue = value;
    in->targetValue = target;
    
    //计算误差，误差积分及值微分
    in->error = in->targetValue - in->nowValue;
    in->ierror += in->error * in->dt;
    in->dvalue = (in->nowValue - in->lastValue) / in->dt;

    in->CtlValue = in->Kp * in->error + in->Ki * in->ierror - in->Kd * in->dvalue; //计算控制值

    //输出限制
    if(in->CtlValue > in->CtlMax)
    {
        in->CtlValue = in->CtlMax;
        in->ierror -= in->error * in->dt;//不计此次积分误差，若长时间CtlValue值被截断，积分误差将变得很大，回调会很慢
    }
    if(in->CtlValue < in->CtlMin)
    {
        in->CtlValue = in->CtlMin;
        in->ierror -= in->error * in->dt;//不计此次积分误差
    }

    in->lastValue = in->nowValue;  //更新上次值

    return in-> CtlValue;
}


//设置控制限制
void pid_set_limit(PID *in,double max,double min)
{
    in->CtlMax = max;
    in->CtlMin = min;
}

//释放
void pid_destroy(PID *in)
{
    if(in)
    {
        free(in);
    }
}