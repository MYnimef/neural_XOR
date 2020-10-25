#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <A:\\Coding\\Cpp\\Git\\neural_XOR\\neuralXOR\\ohMy.cpp>

using namespace std;

int my_xor(int x1, int x2);
float neuralNetwork(int x1, int x2, int maxEpoch, int target);
float sigmoidFunction(float x);
float sigmoidPrime(float x);

int main()
{
    bool task = true;
    while (task)
    {
        cout << "This is a nerural network that calculates XOR." << endl <<
            "Enter x1, x2: ";
        int x1, x2;
        cin >> x1 >> x2;

        int target = my_xor(x1, x2);

        cout << "Enter max Epoch value: ";
        int maxEpoch;
        cin >> maxEpoch;
        cout << neuralNetwork(x1, x2, maxEpoch, target) << endl;

        task = mojemPovtorit();
    }
    return 0;
}

int my_xor(int x1, int x2)  //XOR, для того, чтобы знать какой результат должен быть
{
    int result;
    if (x1 == x2)
    {
        result = 0;
    }
    else
    {
        result = 1;
    }
    return result;
}

float neuralNetwork(int x1, int x2, int maxEpoch, int target)
{
    srand(time(0));     //Настройка генерации чисел от времени
    const int n = 3;    //Кол-во скрытых слоев
    float result = 0;   //Выходной результат программы, в случае если кол-во эпох заданное пользователем равно 0

    float w1[n], w2[n], w3[n];
    for (int i = 0; i < n; i++)     //Цикл, присваивающий рандомные значения весам (диапазон от 0.1 до 0.9)
    {
        w1[i] = (1 + rand() % 9) * 0.1;     //Вес от первого входа до скрытого слоя
        w2[i] = (1 + rand() % 9) * 0.1;     //Вес от второго входа до скрытого слоя
        w3[i] = (1 + rand() % 9) * 0.1;     //Вес от скрытого слоя до выхода
    }

    for (int epoch = 0; epoch < maxEpoch; epoch++)   //Начало работы самой нейросети 
    {
        float pre_hiddenLayer[n], hiddenLayer[n], pre_result = 0;   //Расчет значений до скрытого слоя, в скрытом слое и то, что из него выйдет
        for (int i = 0; i < n; i++)
        {
            pre_hiddenLayer[i] = x1 * w1[i] + x2 * w2[i];
            hiddenLayer[i] = sigmoidFunction(pre_hiddenLayer[i]);
            pre_result += hiddenLayer[i] * w3[i];
        }
        result = sigmoidFunction(pre_result);   //Конечный результат

        float margin_of_error = target - result;    //Сравнение результата с ожидаемым ответом и вычисление погрешности
        float deltaOutSum = sigmoidPrime(pre_result) * margin_of_error;     //Погрешность суммы до выхода, использование производной функции преобразования
        float deltaWeights[n], deltaHiddenSum[n];
        for (int i = 0; i < n; i++)
        {
            deltaWeights[i] = deltaOutSum / hiddenLayer[i];
            w3[i] = w3[i] + deltaWeights[i];

            deltaHiddenSum[i] = deltaOutSum / w3[i] * sigmoidPrime(pre_hiddenLayer[i]);
            w1[i] = deltaHiddenSum[i] * x1 + w1[i];
            w2[i] = deltaHiddenSum[i] * x2 + w2[i];
        }
    }
    return result;
}

float sigmoidFunction(float x)  //Функция преобразования внутри нейрона
{
    float f = 1 / (1 + exp(-x));
    return f;
}

float sigmoidPrime(float x)     //Производная этой функции
{
    float f = exp(-x) / pow((1 + exp(-x)), 2);
    return f;
}
