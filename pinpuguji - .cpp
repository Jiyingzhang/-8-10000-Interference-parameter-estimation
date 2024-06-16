#include <stdio.h>

#define ARRAY_SIZE 1024
#define THRESHOLD_DIFF 0.5 // 阈值差值，用于确定周围频点的幅值是否与最高点接近

// 估算干扰带宽的函数
void estimateInterferenceBandwidth(float spectrum[], int arraySize, float Fs) {
    float maxPower = 0.0;
    int maxPowerIndex = 0;

    // 找到功率谱幅值最大值
    for (int i = 0; i < arraySize; i++) {
        if (spectrum[i] > maxPower) {
            maxPower = spectrum[i];
            maxPowerIndex = i;
        }
    }

    // 计算干扰功率最大值的周围幅度平均值
    float sumAroundMax = spectrum[maxPowerIndex];
    int count = 1;
    for (int i = maxPowerIndex - 1; i >= 0; i--) {
        if (spectrum[maxPowerIndex] - spectrum[i] < THRESHOLD_DIFF) {
            sumAroundMax += spectrum[i];
            count++;
        } else {
            break;
        }
    }
    for (int i = maxPowerIndex + 1; i < arraySize; i++) {
        if (spectrum[maxPowerIndex] - spectrum[i] < THRESHOLD_DIFF) {
            sumAroundMax += spectrum[i];
            count++;
        } else {
            break;
        }
    }
    float avgPowerAroundMax = sumAroundMax / count;

    // 计算干扰带宽的阈值（干扰功率最大值下降3dB处）
    float thresholdPower = maxPower - 3.0;

    // 找到干扰带宽的上标
    float maxUpperBandwidth = 0.0;
    for (int i = 0; i < arraySize; i++) {
        if (spectrum[i] > thresholdPower) {
            float freq = (float)i * Fs / arraySize;
            if (freq > maxUpperBandwidth) {
                maxUpperBandwidth = freq;
            }
        }
    }

    printf("干扰带宽的上标: %f Hz\n", maxUpperBandwidth);
}

// 判断梳状类型干扰的个数并估算各个梳状的3dB带宽和中心频率的函数
void estimateCombInterference(float spectrum[], int arraySize, float Fs) {
    int combCount = 0; // 记录梳状干扰的个数

    for (int i = 1; i < arraySize - 1; i++) {
        if (spectrum[i] > spectrum[i-1] && spectrum[i] > spectrum[i+1]) {
            float peakPower = spectrum[i];
            float threshold = peakPower / 2;
            float centerFreq = (float)i * Fs / arraySize;
            float lowFreq = centerFreq;
            float highFreq = centerFreq;

            // 计算3dB带宽
            while (spectrum[(int)(lowFreq * arraySize / Fs)] > threshold && lowFreq > 0) {
                lowFreq -= Fs / arraySize;
            }

            while (spectrum[(int)(highFreq * arraySize / Fs)] > threshold && highFreq < Fs/2) {
                highFreq += Fs / arraySize;
            }

            float bandwidth3dB = highFreq - lowFreq;

            printf("梳状干扰 %d:\n", combCount + 1);
            printf("中心频率: %f Hz\n", centerFreq);
            printf("3dB带宽: %f Hz\n", bandwidth3dB);

            combCount++;
        }
    }

    printf("梳状干扰个数: %d\n", combCount);
}

// 估计频谱的中心频率的函数
float estimateCenterFrequency(float spectrum[], int arraySize) {
    // 步骤1：计算功率谱
    // (假设功率谱已经计算好)

    // 步骤2：找到功率谱中的最大值
    float maxValue = 0;
    for (int i = 0; i < arraySize; i++) {
        if (spectrum[i] > maxValue) {
            maxValue = spectrum[i];
        }
    }

    // 步骤3：计算阈值数组
    float thresholdArray[arraySize];
    for (int k = 1; k <= arraySize; k++) {
        thresholdArray[k-1] = maxValue / k;
    }

    // 步骤4：找到大于阈值的最小下标和最大下标，计算中心频率的估计值
    float centerFrequency = 0;
    for (int j = 0; j < arraySize; j++) {
        if (spectrum[j] >= thresholdArray[0]) {
            int minIndex = j;
            int maxIndex = j;

            for (int m = 1; m < arraySize; m++) {
                if (spectrum[j] < thresholdArray[m]) {
                    maxIndex = j - 1;
                    break;
                }
            }

            centerFrequency = (float)(minIndex + maxIndex) / 2;
            break;
        }
    }

    return centerFrequency;
}