using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows;
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

            TypeIntsList.Add(new TypeInts() { Name = "ha", ValueString = "1,2,3" });
        }

        

        public List<TypeInts> TypeIntsList = new List<TypeInts>();

        public double[][] Instances; //data

        

        private void DirectoryButton_Click(object sender, RoutedEventArgs e)
        {
            
            if(String.IsNullOrWhiteSpace(numInputBox.Text) || String.IsNullOrWhiteSpace(numNodeBox.Text) || String.IsNullOrWhiteSpace(numOutputBox.Text))
            {
                MessageBox.Show("Please enter the number of inputs, nodes, and outputs");
                return;
            }

            if (separatorBox.Text.Count() != 1)
            {
                MessageBox.Show("Please make sure the separator is a single character");
                return;
            }

            foreach (TypeInts i in TypeIntsList)
                foreach(char j in i.ValueString)
                    if (!(char.IsDigit(j) || j.Equals(',')))
                    {
                        MessageBox.Show("Please make sure int[] values only contain integers and commas");
                        return;
                    }



            OpenFileDialog fileDialog = new OpenFileDialog()
            {
                Filter = "All Files (*.*)|*.*",
                FilterIndex = 1
            };

            if (fileDialog.ShowDialog() == true)
            {
                string[] lines = File.ReadAllLines(fileDialog.FileName);

                foreach (string i in lines)
                    foreach (TypeInts j in TypeIntsList)
                        i.Replace(j.Name, j.ValueString);   // Replaces types with int values

                if (lines[0].Split(char.Parse(separatorBox.Text)).Count() != int.Parse(numInputBox.Text) + int.Parse(numOutputBox.Text))
                {
                    MessageBox.Show("Error: data count does not match.\nPlease note that input/output numbers are values AFTER processing categories");
                    return;
                }

                Instances = GetInstances(lines, Char.Parse(separatorBox.Text));

                StartNeuralNetwork();
            }
        }

        double[][] GetInstances(string[] lines, char separator)
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
            MakeTrainTest(Instances, out double[][] trainData, out double[][] testData, TrainPercentSlider.Value);
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

        static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0) Console.WriteLine("");
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
            if (newLine == true) Console.WriteLine("");
        }


    } //public class Mainwindow

    public class TypeInts
    {
        public string Name { get; set; }

        public string ValueString { get; set; }
    }
}
