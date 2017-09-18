using System;

namespace NeuralNetwork
{
    public class NeuralNetworkObject
    {
        private static Random rnd;

        private int numInput;
        private int numHidden;
        private int numLayer;
        private int numOutput;

        private double[] inputs;


        //i = input, o = output, h = hideen

        private double[][] ihWeights; //input-hidden
        private double[][] hBiases;   //[layer][node]
        private double[] hOutputs;

        private double[][][] nodeWeights;   //[layer][node1][node2]

        private double[][] hoWeights; //hidden-output
        private double[] oBiases;

        private double[] outputs;

        // back-prop specific arrays
        private double[] oGrads; //output gradients for back-propagation
        private double[][] hGrads; // hidden gradients for back-propagation

        // back-prop momentum specific arrays, i.e. to know how much too add/subtract
        private double[][] ihPrevWeightsDelta;  // for momentum with back-propagation
        private double[][] hPrevBiasesDelta;
        private double[][] hoPrevWeightsDelta;
        private double[] oPrevBiasesDelta;

        public NeuralNetworkObject(int numInput, int numHidden, int numLayer, int numOutput)
        {
            rnd = new Random(0); // for Initializing weights and shuffling

            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];

            this.ihWeights = MakeMatrix(numInput, numHidden); //Weights from input nodes to hidden nodes
            this.hBiases = MakeMatrix(numLayer, numHidden);
            this.hOutputs = new double[numHidden];

            this.hoWeights = MakeMatrix(numHidden, numOutput); //Weights from hidden nodes to output nodes
            this.oBiases = new double[numOutput];

            this.outputs = new double[numOutput];

            // back-prop related arrays
            this.hGrads = MakeMatrix(numLayer, numHidden);
            this.oGrads = new double[numOutput];

            this.ihPrevWeightsDelta = MakeMatrix(numInput, numHidden);
            this.hPrevBiasesDelta = MakeMatrix(numLayer, numHidden);
            this.hoPrevWeightsDelta = MakeMatrix(numHidden, numOutput);
            this.oPrevBiasesDelta = new double[numOutput];
        }

        private static double[][] MakeMatrix(int rows, int columns)
        {
            double[][] result = new double[rows][];
            for (int r = 0; r < result.Length; ++r)
                result[r] = new double[columns];
            return result;

            //because you can't do
            //new double[rows][columns];
            //for some magical reason
        }


        // ----------------------------------------------------------------------------------------

        public void SetWeights(double[] weights)
        {
            //copy weights and biases in wieghts[] array to i-h weights, i-h biases, h-o weights, h-o biases
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + (numHidden * numLayer) + (int)(Math.Pow(numHidden, numLayer)) + numOutput;
                             //Input to 1st layer      last layer to output      node bias                Node to node (node^layer e.g. 3x3x3)   output bias
            if (weights.Length != numWeights)
                throw new Exception("Bad weights array length: "); //prevents out of bound error

            int k = 0; // points into weights param

            //assigning weights, foreach x -> foreach y -> assign value
            //input -> 1st layer
            for (int i = 0; i < numInput; ++i)
                for (int j = 0; j < numHidden; ++j)
                    ihWeights[i][j] = weights[k++];
            //last layer -> output weights
            for (int i = 0; i < numHidden; ++i)
                for (int j = 0; j < numOutput; ++j)
                    hoWeights[i][j] = weights[k++];
            //node biases
            for (int i = 0; i < numLayer; ++i)
                for (int j = 0; j < numHidden; ++j)
                    hBiases[i][j] = weights[k++];

            //node -> node weights
            for (int i = 0; i < numLayer; ++i)
                for (int j = 0; j < numHidden; ++j)
                    for (int m = 0; k < numHidden; ++k)
                        nodeWeights[i][j][m] = weights[k++];

            //output biases
            for (int i = 0; i < numOutput; ++i)
                oBiases[i] = weights[k++];
        }

        public void InitializeWeights()
        {
            //initialize weights and biases to small random values
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + (numHidden * numLayer) + (int)(Math.Pow(numHidden, numLayer)) + numOutput;
                            //Input to 1st layer      last layer to output      node bias                Node to node (node^layer e.g. 3x3x3)   output bias
            double[] initialWeights = new double[numWeights];
            double lo = -0.01;
            double hi = 0.01;
            for (int i = 0; i < initialWeights.Length; ++i)
                initialWeights[i] = (hi - lo) * rnd.NextDouble() + lo;
            //0.02 * random digit - 0.01

            this.SetWeights(initialWeights);
        }

        public double[] GetWeights()
        {
            //returns the current set of weights, presumably after training
            //basically reversed setweights
            int numWeights = (numInput * numHidden) + (numHidden * numOutput) + (numHidden * numLayer) + (int)(Math.Pow(numHidden, numLayer)) + numOutput;
                            //Input to 1st layer      last layer to output      node bias                Node to node (node^layer e.g. 3x3x3)   output bias
            double[] result = new double[numWeights];
            int k = 0;

            //input -> 1st layer
            for (int i = 0; i < ihWeights.Length; ++i)
                for (int j = 0; j < ihWeights[0].Length; ++j)
                    result[k++] = ihWeights[i][j];
            //last layer -> output layer
            for (int i = 0; i < hoWeights.Length; ++i)
                for (int j = 0; j < hoWeights[0].Length; ++j)
                    result[k++] = hoWeights[i][j];
            //node bias
            for (int i = 0; i < hBiases.Length; ++i)
                for(int j = 0; j < hBiases[0].Length; ++j)
                    result[k++] = hBiases[i][j];
            //node -> node weights  [layer][node1][node2]
            for (int i = 0; i < nodeWeights.Length; ++i)
                for (int j = 0; j < nodeWeights[0].Length; ++j)
                    for (int m = 0; m < nodeWeights[0][0].Length; ++m)
                        result[k++] = nodeWeights[i][j][m];
            //output bias
            for (int i = 0; i < oBiases.Length; ++i)
                result[k++] = oBiases[i];
            return result;
        }

        // ----------------------------------------------------------------------------------------

        private double[] ComputeOutputs(double[] xValues)
        {
            if (xValues.Length != numInput)
                throw new Exception("Bad xValues array length");

            double[][] hSums = MakeMatrix(numLayer, numHidden); // hidden nodes sums scratch array
            double[] oSums = new double[numOutput]; // output nodes sums

            for (int i = 0; i < xValues.Length; ++i) // copy x-values to inputs
                this.inputs[i] = xValues[i];

            for (int j = 0; j < numHidden; ++j)  // compute i-h sum of weights * inputs
                for (int i = 0; i < numInput; ++i)
                    hSums[0][j] += this.inputs[i] * this.ihWeights[i][j]; // first layer of node

            for (int i = 0; i < numHidden; ++i)  // add biases
                hSums[0][i] += this.hBiases[0][i];

            for (int i = 0; i < numHidden; ++i)
                hSums[0][i] = HyperTanFunction(hSums[0][i]); // Gets hypertan

            if (numLayer > 1)
            {
                for(int i = 1; i < numLayer; ++i) //layer starts at 1 because we've already calculated the 1st layer
                {
                    for (int j = 0; j < numHidden; ++j)
                    {
                        for (int m = 0; m < numHidden; ++m) // m represents the nodes from the last layer
                            hSums[i][j] += this.nodeWeights[i - 1][m][j] * hSums[i - 1][m]; // weights * last layer sums
                                                                                            //since nodeweights doesn't count the input to node weights

                        hSums[i][j] += hBiases[i][j];   // adds bias

                        hSums[i][j] = HyperTanFunction(hSums[i][j]);
                    }
                }
            }

            

            for (int j = 0; j < numOutput; ++j)   // compute h-o sum of weights * hOutputs
                for (int i = 0; i < numHidden; ++i)
                    oSums[j] += hOutputs[i] * hoWeights[i][j];

            for (int i = 0; i < numOutput; ++i)  // add biases to input-to-hidden sums
                oSums[i] += oBiases[i];

            double[] softOut = Softmax(oSums); // softmax activation does all outputs at once for efficiency
            Array.Copy(softOut, outputs, softOut.Length);

            double[] retResult = new double[numOutput]; // could define a GetOutputs method instead
            Array.Copy(this.outputs, retResult, retResult.Length);
            return retResult;
        } // ComputeOutputs

        private static double HyperTanFunction(double x)
        {
            if (x < -20.0) return -1.0; // approximation is correct to 30 decimals
            else if (x > 20.0) return 1.0;
            else return Math.Tanh(x);
        }

        private static double[] Softmax(double[] oSums)
        {
            // determine max output sum
            // does all output nodes at once so scale doesn't have to be re-computed each time
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
                if (oSums[i] > max) max = oSums[i];

            // determine scaling factor -- sum of exp(each val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
                scale += Math.Exp(oSums[i] - max); //e^-x

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
                result[i] = Math.Exp(oSums[i] - max) / scale; //scales it to a sum of 1

            return result; // now scaled so that x sum to 1.0
        }

        // ----------------------------------------------------------------------------------------

        private void UpdateWeights(double[] tValues, double learnRate, double momentum, double weightDecay)
        {
            // update the weights and biases using back-propagation, with target values, eta (learning rate),
            // alpha (momentum).
            // assumes that SetWeights and ComputeOutputs have been called and so all the internal arrays
            // and matrices have values (other than 0.0)
            if (tValues.Length != numOutput)
                throw new Exception("target values not same Length as output in UpdateWeights :");

            //1. compute output gradients
            for (int i = 0; i < oGrads.Length; ++i)
            {
                //derivative of softmax = (1 - y) * y (same as log-sigmoid)
                double derivative = (1 - outputs[i]) * outputs[i];
                // 'mean squared error version includes (1-y)(y) derivative
                oGrads[i] = derivative * (tValues[i] - outputs[i]);
            }

            //Hidden gradients add layer

            //2. compute hidden gradients
            for (int i = 0; i < hGrads.Length; ++i)
            {
                for(int k = 0; k < hGrads[0].Length; k++)
                {
                    //derivative of tanh = (1 - y)(1 + y)
                    double derivative = (1 - hOutputs[i]) * (1 + hOutputs[i]);
                    double sum = 0.0;
                    for (int j = 0; j < numOutput; ++j) // each hidden delta is the sum of numOutput terms
                    {
                        double x = oGrads[j] * hoWeights[i][j];
                        sum += x;
                    }
                    hGrads[i][k] = derivative * sum;
                }
                
            }

            // 3a. update hidden weights (gradients must be computed right-to-left but weights
            // can be updated in any order)
            for (int i = 0; i < ihWeights.Length; ++i) // 0..2 (3)
            {
                for (int j = 0; j < ihWeights[0].Length; ++j) // 0..3 (4)
                {
                    double delta = learnRate * hGrads[j] * inputs[i]; // compute the new delta
                    ihWeights[i][j] += delta; // update. note we use '+' instead of '-'. this can be very tricky.
                                              // now add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
                    ihWeights[i][j] += momentum * ihPrevWeightsDelta[i][j];
                    ihWeights[i][j] -= (weightDecay * ihWeights[i][j]); // weight decay
                    ihPrevWeightsDelta[i][j] = delta; // don't forget to save the delta for momentum 
                }
            }

            // 3b. update hidden biases
            // i = layer        j = node
            for (int i = 0; i < hBiases.Length; ++i)
            {
                for(int j = 0; j < hBiases[0].Length; ++j)
                {
                    double delta = learnRate * hGrads[i] * 1.0; // t1.0 is constant input for bias; could leave out
                    hBiases[i][j] += delta;
                    hBiases[i][j] += momentum * hPrevBiasesDelta[i][j]; // momentum
                    hBiases[i][j] -= (weightDecay * hBiases[i][j]); // weight decay
                    hPrevBiasesDelta[i][j] = delta; // don't forget to save the delta
                }
                
            }

            // 4. update hidden-output weights
            for (int i = 0; i < hoWeights.Length; ++i)
            {
                for (int j = 0; j < hoWeights[0].Length; ++j)
                {
                    // see above: hOutputs are inputs to the nn outputs
                    double delta = learnRate * oGrads[j] * hOutputs[i];
                    hoWeights[i][j] += delta;
                    hoWeights[i][j] += momentum * hoPrevWeightsDelta[i][j]; // momentum
                    hoWeights[i][j] -= (weightDecay * hoWeights[i][j]); // weight decay
                    hoPrevWeightsDelta[i][j] = delta; // save
                }
            }

            // 4b. update output biases
            for (int i = 0; i < oBiases.Length; ++i)
            {
                double delta = learnRate * oGrads[i] * 1.0;
                oBiases[i] += delta;
                oBiases[i] += momentum * oPrevBiasesDelta[i]; // momentum
                oBiases[i] -= (weightDecay * oBiases[i]); // weight decay
                oPrevBiasesDelta[i] = delta; // save
            }

        } // UpdateWeights

        // ----------------------------------------------------------------------------------------

        public void Train(double[][] trainData, int maxEprochs, double learnRate, double momentum, double weightDecay, double exitError)
        {
            // train a back-prop style NN classifier using learning rate and momentum
            // weight decay reduces the magnitude of a weight value over time unless that value
            // is constantly increased

            int epoch = 0;
            double[] xValues = new double[numInput]; //inputs
            double[] tValues = new double[numOutput]; //target values

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
                sequence[i] = i;                        //0,1,2,3,4,5,6,7,8....

            while (epoch < maxEprochs)
            {
                double mse = MeanSquaredError(trainData);
                if (mse < exitError) break; //train until mse reaches the goal exitError

                Shuffle(sequence);
                for (int i = 0; i < trainData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValues, numInput); //copies input data
                    Array.Copy(trainData[idx], numInput, tValues, 0, numOutput); //copies output data
                    ComputeOutputs(xValues); // copy xValues in, compute outputs (store them internally)
                    UpdateWeights(tValues, learnRate, momentum, weightDecay); // find better weights
                } // each training tuple
                ++epoch;
            }
        } // Train

        private static void Shuffle(int[] sequence)
        {
            //randomly swap ints around to create a random sequence
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }
        }

        private double MeanSquaredError(double[][] trainData)
        {
            //avaerage squared error per training tuple
            double sumSquaredError = 0.0;
            double[] xValues = new double[numInput]; // first numInput values in trainData
            double[] tValues = new double[numOutput]; // last numOutput values

            //loop through each training case. looks like (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target values
                double[] yValues = this.ComputeOutputs(xValues); // compute output using current weights
                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - yValues[j];
                    sumSquaredError += err * err;
                }
            }

            return sumSquaredError / trainData.Length;
        }

        // ----------------------------------------------------------------------------------------

        public double Accuracy(double[][] testData)
        {
            // percentage correct using winner-takes all
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput]; // inputs
            double[] tValues = new double[numOutput]; // targets
            double[] yValues; // computed Y

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput); // parse test data into x-values and t-values
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = MaxIndex(yValues); // which cell in yValues has largest value?

                if (tValues[maxIndex] == 1.0) // ugly. consider AreEqual(double x, double y)
                    ++numCorrect;
                else
                    ++numWrong;
            }
            return (numCorrect * 1.0) / (numCorrect + numWrong); // ugly 2 - check for divide by zero
        }

        private static int MaxIndex(double[] vector) // helper for Accuracy()
        {
            // index of largest value
            int bigIndex = 0;
            double biggestVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > biggestVal)
                {
                    biggestVal = vector[i]; bigIndex = i;
                }
            }
            return bigIndex;
        }

    } // NeuralNetwork
}
