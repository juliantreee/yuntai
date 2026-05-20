#ifndef PID_H_
#define PID_H_


typedef struct PID PID;

PID* pid_create(double kp, double ki, double kd, double Ctlmax, double Ctlmin,double dt); //PID对象创建函数
double pid_step(PID *in,double value,double target);  //单次pid函数
void pid_clear(PID *in); //重置pid
void pid_set_limit(PID *in, double max, double min); //设置输出限制
void pid_destroy(PID *in);//释放内存

#endif