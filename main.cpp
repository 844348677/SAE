#include <iostream>
#include <math.h>
#include <arrayfire.h>
#include <exception>

using namespace std;
using namespace af;

//Sparse Autoencoder 抄抄改改，test小程序
//输入数据，按为一个样本，输入4个，隐藏2个，输出4个
float input[5][4]={{0,0,0,1},
                              {0,0,1,0},
                              {0,1,0,0},
                              {1,0,0,0},
                              {1,1,1,1}};
float edge_1[2][5];
float edge_2[4][3];
float first_layer[5];
float hidden_layer[3]={0,0,1};
float output_layer[4];

float d = 0.4;

float function_sigmoid(float x){
    float ex = (float) pow(2.718281828,x);
    return 1/(1+ex);
}

//输出减输入的差的平方的和
float diff(){
    //cout << "diff:"<< endl;
    //af::setBackend(AF_BACKEND_CPU);
    array A(4,1,first_layer); //删除了最后一个1
    array B(4,1,output_layer);

    float result = af::sum<float>(pow(A-B,2));
    return result;
}

//边1 2×5的矩阵与输入层5×1的矩阵相乘，得到2×1的矩阵为隐藏层
void calculate_first(){
    //array A(2,5,)
    float edge_1_tmp_first[5] = {0};
    float edge_1_tmp_second[5]={0};
    for(int i=0;i<5;++i){
        edge_1_tmp_first[i]=edge_1[0][i];
        edge_1_tmp_second[i]=edge_1[1][i];
    }
    array A1(1,5,edge_1_tmp_first);
    array A2(1,5,edge_1_tmp_second);
    array B(1,5,first_layer);

    hidden_layer[0] = function_sigmoid(af::sum<float>(A1*B));
    hidden_layer[1] = function_sigmoid(af::sum<float>(A2*B));
}

//边2 4×3的矩阵与隐藏曾3×1的矩阵相乘，得到4×1的输出层，之后要输出层接近等于隐藏层
void calculate_second(){
    float edge_2_tmp_first[3]={0};
    float edge_2_tmp_second[3]={0};
    float edge_2_tmp_third[3]={0};
    float edge_2_tmp_fourth[3]={0};

    for(int i=0;i<3;++i){
        edge_2_tmp_first[i] = edge_2[0][i];
        edge_2_tmp_second[i] = edge_2[1][i];
        edge_2_tmp_third[i] = edge_2[2][i];
        edge_2_tmp_fourth[i] = edge_2[3][i];
    }

    array A1(1,3,edge_2_tmp_first);
    array A2(1,3,edge_2_tmp_second);
    array A3(1,3,edge_2_tmp_third);
    array A4(1,3,edge_2_tmp_fourth);
    array B(1,3,hidden_layer);

    output_layer[0] = function_sigmoid(af::sum<float>(A1*B));
    output_layer[1] = function_sigmoid(af::sum<float>(A2*B));
    output_layer[2] = function_sigmoid(af::sum<float>(A3*B));
    output_layer[3] = function_sigmoid(af::sum<float>(A4*B)) ;
}
//整个矩阵的输入输出的差的平方的和
float total_diff(){
    float ret = 0;
    for(int i=0;i<4;++i){
        first_layer[0] = input[i][0];
        first_layer[1] = input[i][1];
        first_layer[2] = input[i][2];
        first_layer[3] = input[i][3];
        first_layer[4] = input[i][4];
        calculate_first();
        calculate_second();
        ret = ret + diff();

    }
    return ret;
}

void sparseAE(){
    cout << "sparseAE:" << endl;
    //diff();
    //float result = total_diff();
    //cout << result << endl;

    //初始化边1
    for(int i=0;i<2;++i)
        for(int j=0;j<5;++j){
            edge_1[i][j] = ((i+j)%4+0.1)/10;
            }

     //初始化边2
    for(int i=0;i<4;++i)
        for(int j=0;j<3;++j){
            edge_2[i][j] = ((i+j)%4+0.1)/10;
            }

    //train iterate 40000
    for(int i=0;i<40000;++i){
        float origin_diff = total_diff();

        float direction_1[2][5],direction_2[4][3];
        for(int i=0;i<2;++i){
            for(int j=0;j<5;++j){
                float tmp = edge_1[i][j];
                edge_1[i][j] += d*origin_diff;
                calculate_first();
                calculate_second();
                float diff2 = total_diff();
                direction_1[i][j] = origin_diff - diff2;
                edge_1[i][j] = tmp;
            }
        }
        for(int i=0;i<4;++i){
            for(int j=0;j<3;++j){
                float tmp = edge_2[i][j];
                edge_2[i][j] += d*origin_diff;
                calculate_first();
                calculate_second();
                float diff2 = total_diff();
                direction_2[i][j] = origin_diff - diff2;
                edge_2[i][j] =tmp;
            }
        }

        for(int i=0;i<2;++i){
            for(int j=0;j<5;++j){
                edge_1[i][j] += d*origin_diff*direction_1[i][j];
            }
        }
        for(int i=0;i<4;++i){
            for(int j=0;j<3;++j){
                edge_2[i][j] += d*origin_diff*direction_2[i][j];
            }
        }
        float d = 0;
        for(int i=0;i<4;++i){
            first_layer[0] = input[i][0];
            first_layer[1] = input[i][1];
            first_layer[2] = input[i][2];
            first_layer[3] = input[i][3];
            first_layer[4] = input[i][4];
            calculate_first();
            calculate_second();
            d += diff();
                cout << "#####################" << endl;
                cout <<"input: " << first_layer[0]<< " " <<first_layer[1] << " " << first_layer[2] << " " << first_layer[3] << endl;
                cout <<"output: " << output_layer[0]<< " " <<output_layer[1] << " " << output_layer[2] << " " << output_layer[3] << endl;
                cout <<"hidden: "<< hidden_layer[0] << " " << hidden_layer[1] << endl;
                cout << "#####################" << endl;
        }
        cout << "diff_sum: " << d << endl;

    }

}
int main3()
{
    sparseAE();
    return 0;
}
