using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;

namespace NeuralNetwork
{
    class Network
    {
        public double LearningRate { get; set; } = 0.1;

        public double Epochs { get; set; } = 200;

        public double DisplaySteps { get; set; } = 10;

        public int BatchSize { get; set; } = 128;

        public int[] NodeSize { get; set; }
        // .length = layers


        public void Train(double[][] input, double[][] output)
        {
            int InputSize = input[0].Length;
            int OutputSize = output[0].Length;

            using (var graph = new TFGraph())
            {

                TFOutput X = graph.Placeholder(TFDataType.Float, new TFShape(new long[] { -1, InputSize }));
                TFOutput Y = graph.Placeholder(TFDataType.Float, new TFShape(new long[] { -1, OutputSize }));

                TFOutput[] weights = new TFOutput[NodeSize.Length + 1];
                TFOutput[] biases = new TFOutput[NodeSize.Length + 1];

                int prevSize = InputSize;
                for(int i = 0; i < NodeSize.Length; i++)
                {
                    weights[i] = graph.VariableV2(new TFShape(new long[] { prevSize, NodeSize[i]  }), TFDataType.Float, operName: "weight_" + i);
                    biases[i] = graph.VariableV2(new TFShape(new long[] { NodeSize[i] }), TFDataType.Float, operName: "bias_" + i);
                    prevSize = NodeSize[i];
                }

                weights[NodeSize.Length] = graph.VariableV2(new TFShape(new long[] { prevSize, OutputSize }), TFDataType.Float, operName: "weight_out");
                biases[NodeSize.Length] = graph.VariableV2(new TFShape(new long[] { OutputSize }), TFDataType.Float, operName: "bias_out");

                TFOutput pred = Predict(X, weights, biases, graph);

                TFOutput cost = graph.ReduceMean(graph.SigmoidCrossEntropyWithLogits(Y, pred));

                TFOutput[] parameters = new TFOutput[weights.Length + biases.Length];
                weights.CopyTo(parameters, 0);
                biases.CopyTo(parameters, weights.Length);

                TFOutput[] grad = graph.AddGradients(new TFOutput[] { cost }, parameters);

                TFOperation[] optimize = new TFOperation[parameters.Length];

                for(int i = 0; i < parameters.Length; i++)
                {
                    optimize[i] = graph.AssignSub(parameters[i], graph.Mul(grad[i], graph.Const(LearningRate))).Operation;
                }

                TFSession sess = new TFSession(graph);

                TFOperation[] InitParams = new TFOperation[parameters.Length];

                for (int i = 0; i < parameters.Length; i++)
                {
                    InitParams[i] = graph.Assign(parameters[i], graph.RandomNormal(graph.GetTensorShape(parameters[i]))).Operation;
                }

                sess.GetRunner().AddTarget(InitParams);

                for (int i = 0; i < Epochs; i++)
                {
                    TFTensor result = sess.GetRunner()
                        .AddInput(X, input)
                        .AddInput(Y, output)
                        .AddTarget(optimize)
                        .Fetch(cost)
                        .Run();

                    if (i % DisplaySteps == 0)
                        Console.WriteLine("Epoch - " + i + " | Cost - " + result.GetValue());
                }
            }

        }

        public TFOutput Predict(TFOutput x, TFOutput[] w, TFOutput[] b, TFGraph graph)
        {
            TFOutput LayerOut = x;

            for(int i = 0; i < w.Length; i++)
            {
                LayerOut = graph.Add(graph.MatMul(LayerOut, w[i]), b[i]);
            }

            return LayerOut;
        }

    }
}
