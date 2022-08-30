#include <iostream>
#include "VectorSumC.h"
#include "MatrixSumC.h"
#include "VectorSumOpenMP.h"
#include "MatrixSumOpenMP.h"
#include "DocSearchOpenMP.h"
#include "DocSearchC.h"

int main()
{
    int trials = 1000;
    int vecSize = 1000000; //number of elements of vector
    int matrixDim = 8; //width and height are the same
    

    ///C
    //vectorSumC(vecSize, trials);

    //matrixSumCOneLoop(matrixDim, matrixDim, trials);
    //matrixSumCTwoLoops(matrixDim, matrixDim, trials);

    matrixSumCOneLoop(1024, 1024, trials);
    matrixSumCOneLoop(2048, 2048, trials);
    matrixSumCOneLoop(4096, 4096, trials);

    matrixSumCTwoLoops(1024, 1024, trials);
    matrixSumCTwoLoops(2048, 2048, trials);
    matrixSumCTwoLoops(4096, 4096, trials);

    //docSearchC(256, 65536, 1000);


    ///OpenMP
    //vectorSumOpenMP(vecSize, trials, 2);
 
    //matrixSumOpenMPOneLoop(matrixDim, matrixDim, trials, 8);
    //matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(matrixDim, matrixDim, trials, 8);
    //matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(matrixDim, matrixDim, trials, 8);
    //matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(matrixDim, matrixDim, trials, 8);
    
    //
    matrixSumOpenMPOneLoop(1024, 1024, trials, 1);
    matrixSumOpenMPOneLoop(1024, 1024, trials, 2);
    matrixSumOpenMPOneLoop(1024, 1024, trials, 4);
    matrixSumOpenMPOneLoop(1024, 1024, trials, 8);

    matrixSumOpenMPOneLoop(2048, 2048, trials, 1);
    matrixSumOpenMPOneLoop(2048, 2048, trials, 2);
    matrixSumOpenMPOneLoop(2048, 2048, trials, 4);
    matrixSumOpenMPOneLoop(2048, 2048, trials, 8);

    matrixSumOpenMPOneLoop(4096, 4096, trials, 1);
    matrixSumOpenMPOneLoop(4096, 4096, trials, 2);
    matrixSumOpenMPOneLoop(4096, 4096, trials, 4);
    matrixSumOpenMPOneLoop(4096, 4096, trials, 8);

    //
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(1024, 1024, trials, 1);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(1024, 1024, trials, 2);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(1024, 1024, trials, 4);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(1024, 1024, trials, 8);

    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(2048, 2048, trials, 1);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(2048, 2048, trials, 2);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(2048, 2048, trials, 4);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(2048, 2048, trials, 8);

    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(4096, 4096, trials, 1);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(4096, 4096, trials, 2);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(4096, 4096, trials, 4);
    matrixSumOpenMPTwoLoopsAndInternalLoopParallelized(4096, 4096, trials, 8);

    //
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(1024, 1024, trials, 1);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(1024, 1024, trials, 2);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(1024, 1024, trials, 4);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(1024, 1024, trials, 8);

    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(2048, 2048, trials, 1);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(2048, 2048, trials, 2);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(2048, 2048, trials, 4);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(2048, 2048, trials, 8);

    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(4096, 4096, trials, 1);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(4096, 4096, trials, 2);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(4096, 4096, trials, 4);
    matrixSumOpenMPTwoLoopsAndExternalLoopParallelized(4096, 4096, trials, 8);

    //
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(1024, 1024, trials, 1);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(1024, 1024, trials, 2);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(1024, 1024, trials, 4);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(1024, 1024, trials, 8);

    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(2048, 2048, trials, 1);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(2048, 2048, trials, 2);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(2048, 2048, trials, 4);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(2048, 2048, trials, 8);

    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(4096, 4096, trials, 1);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(4096, 4096, trials, 2);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(4096, 4096, trials, 4);
    matrixSumOpenMPTwoLoopsAndBothLoopsParallelized(4096, 4096, trials, 8);

    //docSearchOpenMP(128, 1024,1000,8);
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
