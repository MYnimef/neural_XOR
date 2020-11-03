#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <ctime>
#include <cmath>
#include <A:\\Coding\\Cpp\\Git\\neural_XOR\\neuralXOR\\ohMy.cpp>    //Поменяйте себе путь к файлу

using namespace std;

void random_weights(const int n, double w1[], double w2[], double w3[]);
void get_weights(const int n, double w1[], double w2[], double w3[]);
void set_weights(const int n, double w1[], double w2[], double w3[]);
void neural_learning();
double neuralNetwork(int input[]);
double sigmoidFunction(double x);

int main()
{
    cout << "This is a nerural network that calculates XOR." << endl;
    bool task = true;
    while (task)
    {   
        ifstream file_w("weights.txt");
        if (!file_w.is_open())  //если файл не открыт
        {
            neural_learning();
        }
        else
        {
            file_w.close();
        }

        cout << "Enter x1, x2: ";
        int x[2];
        cin >> x[0] >> x[1];

        cout << neuralNetwork(x) << endl;
        task = mojemPovtorit();
    }
    return 0;
}

void random_weights(const int n, double w1[], double w2[], double w3[])
{
    srand(time(0));     //Настройка генерации чисел от времени
    for (int i = 0; i < n; i++)     //Цикл, присваивающий рандомные значения весам (диапазон от -0.5 до 0.5)
    {
        w1[i] = (5.0 - rand() % 10) * 0.1;     //Вес от первого входа до скрытого слоя
        w2[i] = (5.0 - rand() % 10) * 0.1;     //Вес от второго входа до скрытого слоя
        w3[i] = (5.0 - rand() % 10) * 0.1;     //Вес от скрытого слоя до выхода
    }
}

void get_weights(const int n, double w1[], double w2[], double w3[])
{
    ifstream file_w("weights.txt");  //
    for (int i = 0; i < 3 * n; i++)
    {
        string num;
        file_w >> num;
        if (i < 3)
        {
            w1[i] = stof(num);
        }
        else if (i < 6)
        {
            w2[i - 3] = stof(num);
        }
        else
        {
            w3[i - 6] = stof(num);
        }
    }
    file_w.close();
}

void set_weights(const int n, double w1[], double w2[], double w3[])
{
    ofstream file_w("weights.txt");
    for (int i = 0; i < 3 * n; i++)
    {
        if (i < 3)
        {
            file_w << fixed << w1[i] << endl;
        }
        else if (i < 6)
        {
            file_w << fixed << w2[i - 3] << endl;
        }
        else
        {
            file_w << fixed << w3[i - 6] << endl;
        }
    }
    file_w.close();
}

void neural_learning()
{
    const int in = 2;   //Кол-во нейронов на входе
    const int trainSet = 4; //Кол-во тренировачных сетов
    const int n = 3;    //Кол-во нейронов в скрытом слое
    double k = 0.3;  //Коэффициент

    int input[trainSet][in] =   //Значения на входе
    { 
        {0, 0}, 
        {0, 1},
        {1, 0},
        {1, 1} 
    };
    int output[trainSet] =  //Ожидаемый результат работы на данных наборах на входе (XOR)
    { 
        0,
        1,
        1,
        0
    };

    double w_input[in][n], w_output[n]; //Весовые коэффициенты между входом и скрытым слоем и между скрытым слоем и выходом
    random_weights(n, w_input[0], w_input[1], w_output);    //Придание им рандомных значений
    for (int i = 0; i < n; i++)
    {
        cout << w_input[0][i] << " ";
    }
    cout << endl;
    for (int i = 0; i < n; i++)
    {
        cout << w_input[1][i] << " ";
    }
    cout << endl;
    for (int i = 0; i < n; i++)
    {
        cout << w_output[i] << " ";
    }
    cout << endl;
    
    int maxEpoch = 100000; //Кол-во Эпох
    int delta_load = maxEpoch / 10, loading = delta_load;   //Переменные для реализации "загрузки"
    for (int epoch = 0; epoch < maxEpoch; epoch++)   //Начало работы самой нейросети 
    {
        if ((epoch + 1) == loading) //Собственно сама загрузка, 10 звездочек в консоле - финал обучения
        {
            cout << '*';
            loading += delta_load;
        }

        for (int train = 0; train < trainSet; train++)
        {
            //Forward

            double hiddenLayer[n], pre_result = 0;   //Расчет значений до скрытого слоя, в скрытом слое и то, что из него выйдет
            for (int i = 0; i < n; i++)
            {
                double pre_hiddenLayer = 0;
                for (int j = 0; j < in; j++)
                {
                    pre_hiddenLayer += input[train][j] * w_input[j][i];
                }
                hiddenLayer[i] = sigmoidFunction(pre_hiddenLayer);
                pre_result += hiddenLayer[i] * w_output[i];
            }
            double result = sigmoidFunction(pre_result);   //Конечный результат
            //cout << result << endl;   //Чисто для проверки

            //Backward

            double sigma_out = (output[train] - result) * result * (1 - result);     //Погрешность суммы до выхода, использование производной функции преобразования
            double delta_w_output[n], sigma_in[n];
            for (int i = 0; i < n; i++)
            {
                delta_w_output[i] = k * sigma_out * hiddenLayer[i];
                //sigma_in[i] = delta_w_output[i] * w_output[i] * hiddenLayer[i] * (1 - hiddenLayer[i]);        //Тут все еще ведутся
                sigma_in[i] = (output[train] - result) * w_output[i] * hiddenLayer[i] * (1 - hiddenLayer[i]);   //дискуссии

                w_output[i] += delta_w_output[i];
            }

            double delta_w[in][n];
            for (int i = 0; i < in; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    delta_w[i][j] = k * sigma_in[j] * input[train][i];
                    w_input[i][j] += delta_w[i][j];
                }
            }
        }
    }
    cout << endl;
    set_weights(n, w_input[0], w_input[1], w_output);
}

double neuralNetwork(int input[])
{
    const int n = 3;    //Кол-во скрытых слоев
    const int in = 2;
    double w_input[in][n], w_output[n];
    get_weights(n, w_input[0], w_input[1], w_output);

    double hiddenLayer[n], pre_result = 0;
    for (int i = 0; i < n; i++)
    {
        double pre_hiddenLayer = 0;
        for (int j = 0; j < in; j++)
        {
            pre_hiddenLayer += input[j] * w_input[j][i];
        }
        hiddenLayer[i] = sigmoidFunction(pre_hiddenLayer);
        pre_result += hiddenLayer[i] * w_output[i];
    }
    return sigmoidFunction(pre_result);
}

double sigmoidFunction(double x)  //Функция преобразования внутри нейрона
{
    return 1 / (1 + exp(-x));
}