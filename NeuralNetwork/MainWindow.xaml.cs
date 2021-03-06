﻿using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace NeuralNetwork
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
            TypeToIntGrid.ItemsSource = TypeIntsList;
            
        }

        

        public List<TypeInts> TypeIntsList = new List<TypeInts>();

        public double[][] Instances; //data

        public int numInputs = 0;
        public int numOutputs = 0;
        public int numNodes = 0;
        public int numLayers = 0;

        public int MaxEpochs = 0;
        public double LearnRate = 0;
        public double Momentum = 0;
        public double WeightDecay = 0;
        public double ExitError = 0;

        public string[] rawData;

        private void DirectoryButton_Click(object sender, RoutedEventArgs e)
        {

            OpenFileDialog fileDialog = new OpenFileDialog()
            {
                Filter = "All Files (*.*)|*.*",
                FilterIndex = 1
            };

            if (fileDialog.ShowDialog() == true)
            {
                rawData = File.ReadAllLines(fileDialog.FileName);

                if(rawData.Count() == 0)
                {
                    MessageBox.Show("file does not contain any information");
                    return;
                }


                GetStringOptions(rawData, ',');
                Log("Loaded data.");
            }
        }

        void GetStringOptions(string[] lines, char separator)
        {
            string[][] splitedData = new string[lines.Length][]; //2 dimension array, total data * features

            for (int i = 0; i < lines.Length; i++)
            {
                splitedData[i] = lines[i].Split(separator);
            }

            List<int> stringColumns  = new List<int>();
            

            for (int i = 0; i < splitedData[0].Count(); i++)
            {
                if(!double.TryParse(splitedData[0][i], out double value)) //if value does not parse as double
                    stringColumns.Add(i);
            }

            Console.WriteLine(stringColumns[0]);

            foreach (int i in stringColumns)
            {
                int options = 0;
                int startingIndex = TypeIntsList.Count(); //index of the first item

                for (int j = 0; j < splitedData.Length; j++)
                {
                    if(!TypeIntsList.Any(x => x.Name == splitedData[j][i]))     // adds value to list, also counts how many total options there are
                    {
                        TypeIntsList.Add(new TypeInts() { Name = splitedData[j][i], ValueString = "", Index = i });
                        options++;
                        Console.WriteLine(splitedData[j][i]);
                    }
                }

                for(int option = 0; option < options; option++)     //generates an int array with a single 1 to activate different inputs
                {
                    int[] test = new int[options];
                    test[option] = 1;
                    TypeIntsList[startingIndex + option].ValueString = string.Join(",", test);
                }
            }



            TypeToIntGrid.Items.Refresh();  //refreshes the grid (new items won't show without this)
        }

        static double [][] GetInstances(string[] lines, char separator)
        {

            double[][] result = new double[lines.Count()][]; //2 dimension array, total data * features

            for (int i = 0; i < lines.Count(); ++i)
            {
                string[] items = lines[i].Split(separator);
                result[i] = new double[items.Count()];
                for (int j = 0; j < items.Count(); ++j)
                    result[i][j] = double.Parse(items[j]);
            }

            return result;
        }

        private void StartNeuralNetwork()       //Assuming that GetInstances has already run and Instance has value
        {
            double SliderValue = 0;

            Dispatcher.Invoke(() =>
            {

                numInputs = int.Parse(numInputBox.Text);
                numOutputs = int.Parse(numOutputBox.Text);
                numNodes = int.Parse(numNodeBox.Text);
                numLayers = int.Parse(numLayerBox.Text);

                MaxEpochs = int.Parse(MaxEpochBox.Text);
                LearnRate = double.Parse(LearnRateBox.Text);
                Momentum = double.Parse(MomentumBox.Text);
                WeightDecay = double.Parse(WeightDecayBox.Text);
                ExitError = double.Parse(ExitErrorBox.Text);

                SliderValue = TrainPercentSlider.Value;

                Log("Getting started...");

                Log("Separating train and test data.");
            });
            
            MakeTrainTest(Instances, out double[][] trainData, out double[][] testData, SliderValue / 100);

            Dispatcher.Invoke(() => Log("Normalizing all input data"));
            int[] inputs = Enumerable.Range(0, numInputs).ToArray(); //Creates int[] that generates 0,1,2...numInput -1 as numInput is the count

            Normalize(trainData, inputs);
            Normalize(testData, inputs);        //Quick note: maybe we could normalize the data first and then separate them...?

            Dispatcher.Invoke(() => Log("Creating neural network"));

            int[] Layers = new int[numLayers];
            for (int i = 0; i < Layers.Length; i++)
                Layers[i] = numNodes;

            Network net = new Network();
            net.NodeSize = Layers;

            double[][] input = new double[trainData.Length][];
            double[][] outputs = new double[trainData.Length][];

            for (int i = 0; i < trainData.Length; ++i)
            {
                input[i] = new double[numInputs];
                outputs[i] = new double[numOutputs];
                Array.Copy(trainData[i], input[i], numInputs); //copies input data
                Array.Copy(trainData[i], numInputs, outputs[i], 0, numOutputs); //copies output data
            }

            net.Train(input, outputs);

           // NeuralNetworkObject nn = new NeuralNetworkObject(numInputs, numOutputs, Layers);

           // Dispatcher.Invoke(() => Log("Initializing weights"));
           // nn.InitializeWeights();



           // Dispatcher.Invoke(() => Log("Training..."));
           // nn.Train(trainData, MaxEpochs, LearnRate, Momentum, WeightDecay, ExitError);
           // Dispatcher.Invoke(() => Log("Training complete"));

           // double[] weights = nn.GetWeights();
           //Dispatcher.Invoke(() => {
           //    Log("Final Weights and values: ");
           //    ShowVector(weights, 10, 3, true);
           //});

           // double trainAcc = nn.Accuracy(trainData);
           // Dispatcher.Invoke(() => Log("Accuracy on training data = " + trainAcc.ToString("F4")));

           // double testAcc = nn.Accuracy(testData);
           // Dispatcher.Invoke(() => Log("Accuracy on test data = " + testAcc.ToString("F4")));

            Console.WriteLine("Done");
        }


        private void NumericalOnly(object sender, TextCompositionEventArgs e)
        {
            if (e.Text.Length != 1) return;
            e.Handled = !Char.IsDigit(Char.Parse(e.Text));
        }

        private void ColumnNumericalOnly(object sender, TextCompositionEventArgs e)
        {
            if (e.Text.Length != 1) return;
            e.Handled = !(String.Equals(e.Text, ",") || Char.IsDigit(Char.Parse(e.Text)));
        }

        private void DoubleOnly(object sender, TextCompositionEventArgs e)
        {
            if (e.Text.Length != 1) return;
            e.Handled = !(String.Equals(e.Text, ".") || Char.IsDigit(Char.Parse(e.Text)));
        }


        static void MakeTrainTest(double[][] allData, out double[][] trainData, out double[][] testData, double trainPercent)
        {

            Random randomnum = new Random(0);
            int Rows = allData.Length;
            int Columns = allData[0].Length;

            int trainRows = (int)(Rows * trainPercent);
            int testRows = Rows - trainRows;

            trainData = new double[trainRows][];
            testData = new double[testRows][];

            int[] sequence = new int[Rows];
            for (int i = 0; i < sequence.Length; ++i) //initalizes sequence
                sequence[i] = i;

            for (int i = 0; i < sequence.Length; ++i) //swaps numbers to create a random sequence
            {
                int r = randomnum.Next(i, sequence.Length);
                int tmp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = tmp;
            }

            int si = 0; // sequence index
            int j = 0; // index into trainData or testData

            for (; si < trainRows; ++si) //copies data to trainData using random indexes
            {
                trainData[j] = new double[Columns];
                int idx = sequence[si];
                Array.Copy(allData[idx], trainData[j], Columns);
                ++j;
            }

            j = 0; //reset to start of test data

            for (; si < Rows; ++si) //copies data to testData using random indexes
            {
                testData[j] = new double[Columns];
                int idx = sequence[si];
                Array.Copy(allData[idx], testData[j], Columns);
                ++j;
            }

        } // MakeTrainTest

        static void Normalize(double[][] dataMatrix, int[] columns)
        {
            //normalizes values by (x - average) / sd
            foreach (int column in columns)
            {
                double sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += dataMatrix[i][column];
                double average = sum / dataMatrix.Length;

                //sd part - standard deviation http://www.mathsisfun.com/data/standard-deviation.html
                sum = 0.0;
                for (int i = 0; i < dataMatrix.Length; ++i)
                    sum += (dataMatrix[i][column] - average) * (dataMatrix[i][column] - average);
                double sd = Math.Sqrt(sum / (dataMatrix.Length - 1));
                for (int i = 0; i < dataMatrix.Length; ++i)
                    dataMatrix[i][column] = (dataMatrix[i][column] - average) / sd;
            }
        } // Normalize

        void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) ConsoleTextbox.Text += "\n";
                ConsoleTextbox.Text += vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ";
            }
            if (newLine == true) ConsoleTextbox.Text += "\n";
        }

        private void Log(string text)
        {
            string line = string.Format("{0:HH:mm:ss.fff}: {1}\r\n", DateTime.Now, text);
            ConsoleTextbox.AppendText(line);
        }

        private void StartButton_Click(object sender, RoutedEventArgs e)
        {


            if (String.IsNullOrWhiteSpace(numInputBox.Text) || String.IsNullOrWhiteSpace(numNodeBox.Text) || String.IsNullOrWhiteSpace(numOutputBox.Text))
            {
                MessageBox.Show("Please enter the number of inputs, nodes, and outputs");
                return;
            }

            if (separatorBox.Text.Count() != 1)
            {
                MessageBox.Show("Please make sure the separator is a single character");
                return;
            }

            for (int i = 0; i < rawData.Count(); i++)
                foreach (TypeInts j in TypeIntsList)
                {
                    rawData[i] = rawData[i].Replace(j.Name, j.ValueString);   // Replaces types with int values
                }

            if (rawData[0].Split(char.Parse(separatorBox.Text)).Count() != int.Parse(numInputBox.Text) + int.Parse(numOutputBox.Text))
            {
                MessageBox.Show("Error: data count does not match.\nPlease note that input/output numbers are values AFTER processing categories");
                return;
            }

            

            Instances = GetInstances(rawData, Char.Parse(separatorBox.Text));

            Console.WriteLine("HA");

            ThreadStart NeuralNetworkThreadStart = (StartNeuralNetwork);


            Thread NeuralNetworkThread = new Thread(NeuralNetworkThreadStart);

            NeuralNetworkThread.Start();

        }
    } //public class Mainwindow

    public class TypeInts
    {
        public string Name { get; set; }

        public string ValueString { get; set; }

        public int Index { get; set; }
    }
}
