#include <iostream>
#include <math.h>
#include <arrayfire.h>
/*
using namespace std;
using namespace af;

//网络结构
const int input_x_size = 1000; //x的样本个数
const int hidden_layer_size = 3; //有三个隐藏层
const int hidden_first_layer_size = 100; //隐藏层第一层神经元个数
const int hidden_second_layer_size = 200; //隐藏层第二层神经元个数
const int hidden_third_layer_size = 50; //隐藏层第三层神经元个数
const int output_y_size = 1;

//定义网络内容
//b
float b0 = 1;
float b1 = 1;
float b2 = 1;
float b3 = 1;
//输入层
float input_x[input_x_size];
array input_x_array(input_x_size+1,1,input_x);
//隐藏层
array hidden_first_layer =  constant(0,hidden_first_layer_size+1);
array hidden_second_layer =  constant(0,hidden_second_layer_size+1);
array hidden_third_layer =  constant(0,hidden_third_layer_size+1);
//输出层
array output_y = constant(0,output_y_size);
//w
array W1 = constant(0,input_x_size+1,hidden_first_layer_size+1);
array W2 = constant(0,hidden_first_layer_size+1,hidden_second_layer_size+1);
array W3 = constant(0,hidden_second_layer_size+1,hidden_third_layer_size+1);
array W4 = constant(0,hidden_third_layer_size+1,output_y_size);
//函数 h(x)
float fun_logistic(float x){
    float ex = (float) pow(2.718281828,x);
    return 1/(1+ex);
}
float fun_linear(float x){
    return x;
}
float fun_tanh(float x){
    return tanh(x);
}
float fun_nonnegative(float x){
    if(x>0)
        return x;
    else
        return 0;
}
float h(float x){
    return fun_tanh(x);  //   fun_logistic     fun_tanh      fun_linear     fun_nonnegative
}
//逐步 计算  隐藏层
void calculate_hidden_first(){
    for(int i=0;i<hidden_first_layer_size;++i){
        hidden_first_layer(i,0) = h(af::sum<float>(input_x_array*W1.col(i)));
    }
}
void calculate_hidden_second(){
    for(int i=0;i<hidden_second_layer_size;++i){
        hidden_second_layer(i,0) = h(af::sum<float>(hidden_first_layer*W2.col(i)));
    }
}
void calculate_hidden_third(){
    for(int i=0;i<hidden_third_layer_size;++i){
        hidden_third_layer(i,0) = h(af::sum<float>(hidden_second_layer*W3.col(i)));
    }
}
void calculate_output(){
        output_y(0) = h(af::sum<float>(hidden_third_layer*W4.col(0)));
}

//神经元计算主调用函数
void neuronAction(){
    input_x_array(input_x_size,0) = b0;
    hidden_first_layer(hidden_first_layer_size,0) = b1;
    hidden_second_layer(hidden_second_layer_size,0) = b2;
    hidden_third_layer(hidden_third_layer_size,0) = b3;

    calculate_hidden_first();
    calculate_hidden_second();
    calculate_hidden_third();
    calculate_output();
    af_print(output_y);
}

int main(){
    neuronAction();
}
*/
