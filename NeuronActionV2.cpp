#include <iostream>
#include <math.h>
#include <arrayfire.h>

using namespace std;
using namespace af;

//网络结构
const int input_x_size = 1000; //x的样本个数
const int hidden_layer_size = 3; //有三个隐藏层
const int hidden_layers_size[] = {100,200,50}; //隐藏层第一层神经元个数 隐藏层第二层神经元个数 隐藏层第三层神经元个数
const int output_y_size = 1;

//定义网络内容
//b
float bs[hidden_layer_size+1] = {1,1,1,1}; //第一个对应的是输入层，后面对应的隐藏层
//输入层
float input_x[input_x_size];
array input_x_array(input_x_size+1,1,input_x);
//隐藏层
array hidden_layers[hidden_layer_size];
//输出层
array output_y = constant(0,output_y_size);
//w
array Ws[hidden_layer_size+1];
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
float h(float x,int index){
    if(index == 0)
        return fun_logistic(x);     //   fun_logistic     fun_tanh      fun_linear     fun_nonnegative
    if(index == 1)
        return fun_linear(x);
    if(index == 2)
        return fun_tanh(x);
    if(index == 3)
        return fun_nonnegative(x);
    else
        return 0;
}
//逐步 计算  隐藏层
void calculate_hidden_all(){
    for(int i=0;i<hidden_layers_size[0];++i){
        hidden_layers[0](i,0) = h(af::sum<float>(input_x_array*Ws[0].col(i)),0);
    }
    for(int j=0;j<hidden_layer_size-1;j++){
        for(int i=0;i<hidden_layers_size[j+1];++i){
            hidden_layers[j+1](i,0) = h(af::sum<float>(hidden_layers[j]*Ws[j+1].col(i)),0);
        }
    }
    for(int i=0;i<output_y_size;++i){
        output_y(i) = h(af::sum<float>(hidden_layers[hidden_layer_size-1]*Ws[hidden_layer_size].col(i)),0);
    }

}
void init(){
    //初始化各个隐藏层
    for(int i=0;i<hidden_layer_size;++i){
        hidden_layers[i] =constant(0,hidden_layers_size[i]+1);
    }
    //初始化bs
    input_x_array(input_x_size,0) = bs[0];
    for(int i=0;i<hidden_layer_size;++i){
        hidden_layers[i](hidden_layers_size[i],0) = bs[i+1];
    }
    //初始化Ws
    Ws[0] = constant(0,input_x_size+1,hidden_layers_size[0]+1);
    for(int i=0;i<hidden_layer_size-1;++i){
        Ws[i+1] = constant(0,hidden_layers_size[i]+1,hidden_layers_size[i+1]+1);
    }
    Ws[hidden_layer_size] = constant(0,hidden_layers_size[hidden_layer_size-1]+1,output_y_size);
}
//神经元计算主调用函数
void neuronAction(){
    init();
    calculate_hidden_all();
    af_print(output_y);
}

int main(){
    neuronAction();
}

