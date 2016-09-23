#include <iostream>
#include <math.h>
#include <arrayfire.h>

using namespace std;
using namespace af;


float input_original[20] = {0,0,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,0,0,1};
array input_format(5,4,input_original);
array array_edge1 = constant(0,5,2);
array array_edge2 = constant(0,3,4);
array array_input_layer = constant(0,5);
array array_output_layer = constant(0,4);
float hidden_original[3] = {0,0,1};
array array_hidden_layer(3,1,hidden_original);

float dd = 0.4;

float f_sigmoid(float x){
    float ex = (float) pow(2.718281828,x);
    return 1/(1+ex);
}
//输出减输入的差的平方的和
float array_diff(){
    float result = af::sum<float>(pow(array_input_layer.rows(0,3)-array_output_layer,2));
    return result;
}
//边1 2×5的矩阵与输入层5×1的矩阵相乘，得到2×1的矩阵为隐藏层
void array_edge_first(){
    array_hidden_layer(0,0) = f_sigmoid(af::sum<float>(array_input_layer*array_edge1.col(0)));
    array_hidden_layer(1,0) = f_sigmoid(af::sum<float>(array_input_layer*array_edge1.col(1)));
}
//边2 4×3的矩阵与隐藏曾3×1的矩阵相乘，得到4×1的输出层，之后要输出层接近等于隐藏层
void array_edge_second(){
    array_output_layer(0) = f_sigmoid(af::sum<float>(array_hidden_layer*array_edge2.col(0)));
    array_output_layer(1) = f_sigmoid(af::sum<float>(array_hidden_layer*array_edge2.col(1)));
    array_output_layer(2) = f_sigmoid(af::sum<float>(array_hidden_layer*array_edge2.col(2)));
    array_output_layer(3) = f_sigmoid(af::sum<float>(array_hidden_layer*array_edge2.col(3)));
}
//整个矩阵的输入输出的差的平方的和
float total_array_diff(){
    float ret = 0;
    for(int i=0;i<4;i++){
        array_input_layer = input_format.col(i);
        array_edge_first();
        array_edge_second();
        ret += array_diff();
    }
    return ret;
}
void SparseAE(){
    cout << "SparseAE: " << endl;
    //初始化边1
    for(int i=0;i<5;++i)
        for(int j=0;j<2;++j)
            array_edge1(i,j) = ((i+j)%4+0.1)/10;
    //初始化边2
    for(int i=0;i<3;++i)
        for(int j=0;j<4;++j)
            array_edge2(i,j) = ((i+j)%4+0.1)/10;
    //train iterate 40000
    for(int i=0;i<40000;++i){
        float origin_diff = total_array_diff();
        array direction_1 = constant(0,5,2);
        array direction_2 = constant(0,3,4);
        for(int i=0;i<5;++i){
            for(int j=0;j<2;++j){
                array tmp = array_edge1(i,j);
                array_edge1(i,j) += dd*origin_diff;
                array_edge_first();
                array_edge_second();
                float diff2 = total_array_diff();
                direction_1(i,j) = origin_diff - diff2;
                array_edge1(i,j) = tmp;
            }
        }
        for(int i=0;i<3;++i){
            for(int j=0;j<4;++j){
                array tmp = array_edge2(i,j);
                array_edge2(i,j) += dd*origin_diff;
                array_edge_first();
                array_edge_second();
                float diff2 = total_array_diff();
                direction_2(i,j) = origin_diff - diff2;
                array_edge2(i,j) =tmp;
            }
        }
        array_edge1 += dd*origin_diff*direction_1;
        array_edge2 += dd*origin_diff*direction_2;
        float dd = 0;
        for(int i=0;i<4;++i){
            array_input_layer = input_format.col(i);
            array_edge_first();
            array_edge_second();
            dd += array_diff();
            cout << "#####################" << endl;
            /*cout <<"input: " << array_input_layer(0) << " " << array_input_layer(1) << " " << array_input_layer(2) << " " << array_input_layer(3) << endl;
            cout <<"output: " << array_output_layer(0) << " " <<array_output_layer(1)<< " " << array_output_layer(2) << " " << array_output_layer(3) << endl;
            cout <<"hidden: "<< array_hidden_layer(0)<< " " << array_hidden_layer(1) << endl;
            af_print(array_input_layer);
            af_print(array_output_layer);
            af_print(array_hidden_layer);
            */
            cout << "#####################" << endl;
        }
        cout << "diff_sum: " << dd << endl;
    }

}
int main2()
{
    af::info();
    SparseAE();
    return 0;
}
